#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Infinigen LiDAR Viewer (lean)
# Shows LiDAR PLY frames with simple coloring modes and trajectory.
# Example: python lidar_viewer.py path/to/output_dir --color intensity --view camera --frame 12

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

try:
    from plyfile import PlyData
    _HAVE_PLYFILE = True
except Exception:
    _HAVE_PLYFILE = False


# Small helpers

def robust01(x, lo=2, hi=98):
    # Map to [0,1] using robust percentiles so outliers don't dominate
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    p_lo, p_hi = np.percentile(x, [lo, hi])
    if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
        return np.zeros_like(x)
    y = np.clip((x - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    return y

def infer_frame_number(path):
    # Extract frame number from filenames like 'lidar_frame_0001.ply'
    m = re.search(r"lidar_frame_(\d+)\.(?:npz|ply)", os.path.basename(path))
    return int(m.group(1)) if m else 0

def get_first_present(fields: dict, names, default=None):
    # Return first existing field by name
    for n in names:
        if n in fields and fields[n] is not None:
            return fields[n]
    return default

class LidarViewer:
    COLOR_MODES = ['intensity', 'ring']

    def __init__(self, data_dir, color_mode='material'):
        self.data_dir = Path(data_dir)
        self.color_mode = color_mode if color_mode in self.COLOR_MODES else 'intensity'
        self.view_mode = 'world'       # or 'camera'
        self.show_trajectory = True

        # Load configuration & metadata
        self.cfg  = self._load_json(self.data_dir / 'lidar_config.json', default={})
        self.meta = self._load_json(self.data_dir / 'frame_metadata.json', default={"frames": {}})
        self.traj = self._load_json(self.data_dir / 'trajectory.json', default={})
        self.tum_poses = self._load_tum_poses(self.data_dir / 'poses_tum.txt')

        # Basic config
        self.ply_frame = self.cfg.get('ply_frame', 'camera')  # camera/sensor/world

        # Index frames
        self.frame_files = self._scan_frames()
        self.current_idx = 0
        self.camera_positions = [None] * len(self.frame_files)

        # Open3D handles
        self._vis = None
        self._pcd = None
        self._line_set = None
        self._cam_sphere = None
        self._coord_frame = None

    # IO

    @staticmethod
    def _load_json(path: Path, default=None):
        # Read JSON with safe fallback
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return default

    @staticmethod
    def _load_tum_poses(path: Path):
        # Load TUM-style poses: ts tx ty tz qx qy qz qw
        try:
            poses = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 8:
                        continue
                    ts, tx, ty, tz, qx, qy, qz, qw = map(float, parts)
                    poses.append({'ts': ts, 't': np.array([tx, ty, tz], float), 'q': np.array([qx, qy, qz, qw], float)})
            return poses if poses else None
        except Exception:
            return None

    def _scan_frames(self):
        # Find PLY frames and sort by frame number
        files = sorted(self.data_dir.glob('lidar_frame_*.ply'), key=lambda p: infer_frame_number(str(p)))
        if not files:
            print(f"No lidar_frame_*.ply found in {self.data_dir}")
            sys.exit(1)
        first, last = infer_frame_number(str(files[0])), infer_frame_number(str(files[-1]))
        print(f"Found {len(files)} frame(s). Range: {first}..{last}")
        return files

    def camera_pos_for_frame(self, frame_num):
        # Read camera translation t for this frame from trajectory.json
        try:
            t = self.traj[str(frame_num)]['t']
            return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=float)
        except Exception:
            return np.zeros(3, dtype=float)

    def _scale_used_for_frame(self, frame_num):
        """Scaling used to map raw reflectance → U8 intensity (so we can invert it)."""
        try:
            return float(self.meta['frames'][str(frame_num)]['scale_used'])
        except Exception:
            return 1.0

    # Frame loading

    def load_frame(self, idx):
        if idx < 0 or idx >= len(self.frame_files):
            return None
        path = self.frame_files[idx]
        frame_num = infer_frame_number(str(path))
        cam_pos = self.camera_pos_for_frame(frame_num)
        return self._load_ply(path, frame_num, cam_pos, idx)

    def _load_ply(self, path: Path, frame_num: int, cam_pos: np.ndarray, idx: int):
        # Load PLY with plyfile (preferred) to access custom props like cos_incidence/mat_class
        points = None
        fields = {}
        if _HAVE_PLYFILE:
            try:
                ply = PlyData.read(str(path))
                vtx = ply['vertex']
                # positions
                x = np.asarray(vtx.data['x'], dtype=float)
                y = np.asarray(vtx.data['y'], dtype=float)
                z = np.asarray(vtx.data['z'], dtype=float)
                points = np.vstack([x, y, z]).T
                # Common scalar props
                name_map = {
                    'intensity': 'intensities',
                    'ring': 'ring',
                    'azimuth': 'azimuth_rad',
                    'elevation': 'elevation_rad',
                    'time_offset': 'time_offset',
                    'return_id': 'return_id',
                    'num_returns': 'num_returns',
                    'cos_incidence': 'cos_incidence',
                    'mat_class': 'mat_class',
                    'return_power': 'return_power',
                    'reflectance': 'return_power',  # backward compatibility
                    'range_m': 'range_m',
                    'transmittance': 'transmittance',
                    'exposure_scale': 'transmittance',
                }
                for ply_name, out_name in name_map.items():
                    if ply_name in vtx.data.dtype.names:
                        fields[out_name] = np.asarray(vtx.data[ply_name])
                if all(n in vtx.data.dtype.names for n in ('nx', 'ny', 'nz')):
                    nx = np.asarray(vtx.data['nx'])
                    ny = np.asarray(vtx.data['ny'])
                    nz = np.asarray(vtx.data['nz'])
                    fields['normals'] = np.vstack([nx, ny, nz]).T
            except Exception as e:
                print(f"plyfile failed on {path.name}: {e}. Falling back to Open3D.")
        if points is None:
            # Fallback: geometry only
            pcd = o3d.io.read_point_cloud(str(path))
            points = np.asarray(pcd.points)
            # No aux fields available from Open3D PLY reader.
        self.camera_positions[idx] = cam_pos
        # Distances (respect stored frame)
        fields.setdefault('distances', self.compute_distances(points, cam_pos))
        if 'range_m' not in fields:
            fields['range_m'] = fields['distances']
        return {
            'file_path': str(path),
            'points': points,
            'frame': frame_num,
            'camera_location': cam_pos,
            'fields': fields,
        }

    # Geometry helpers

    def compute_distances(self, points, camera_location):
        # Distance to sensor for camera/sensor-frame clouds, else world distance to camera
        points = np.asarray(points)
        if self.ply_frame in ('camera', 'sensor'):
            return np.linalg.norm(points, axis=1)
        return np.linalg.norm(points - camera_location, axis=1)

    # Material reflectance estimation removed in lean viewer

    # Coloring

    def color_by_mode(self, frame_data, mode):
        pts = frame_data['points']
        fields = frame_data['fields']
        N = len(pts)

        # raw intensity (u8 0..255)
        if mode == 'intensity':
            ints = get_first_present(fields, ['intensities'])
            if ints is None:
                return np.ones((N,3))*0.6, 'Intensity (missing)'
            colors = plt.cm.inferno(robust01(ints))[:, :3]
            return colors, 'Intensity (per-frame scaled U8)'
        # ring (elevation id)
        if mode == 'ring':
            ring = get_first_present(fields, ['ring', 'ring_ids'])
            if ring is None:
                return np.ones((N,3))*0.6, 'Ring (missing)'
            ring = ring.astype(int)
            cmap = plt.cm.tab20
            colors = np.array([cmap(int(r % 20))[:3] for r in ring])
            return colors, 'Ring ID'

        # fallback
        return np.ones((N, 3)) * 0.6, 'Default'

    # Visualization

    def create_geometries(self, frame_data):
        geometries = {}

        # Point cloud + colors
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(frame_data['points'])
        colors, desc = self.color_by_mode(frame_data, self.color_mode)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geometries['pcd'] = pcd
        geometries['color_desc'] = desc

        # Camera marker & axes - make camera more visible
        cam_pos = frame_data['camera_location']
        # Larger sphere with brighter color for better visibility
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        sphere.translate(cam_pos)
        sphere.paint_uniform_color([1.0, 0.2, 0.2])  # Bright red
        geometries['cam_sphere'] = sphere

        # Larger coordinate frame
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=cam_pos)
        geometries['coord_frame'] = axes

        # Trajectory
        if self.show_trajectory:
            idx = self.current_idx
            for j in range(idx + 1):
                if self.camera_positions[j] is None:
                    frm = infer_frame_number(str(self.frame_files[j]))
                    self.camera_positions[j] = self.camera_pos_for_frame(frm)
            pts = np.array([p for p in self.camera_positions[:idx + 1] if p is not None])
            if len(pts) >= 2:
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(pts)
                lines = [[k, k + 1] for k in range(len(pts) - 1)]
                ls.lines = o3d.utility.Vector2iVector(lines)
                ls.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(lines))
                geometries['line_set'] = ls

        return geometries

    def _apply_view_mode(self):
        ctr = self._vis.get_view_control()
        if self.view_mode == 'camera':
            # LiDAR perspective. If PLY is in sensor frame, origin already matches LiDAR.
            if self.ply_frame == 'sensor':
                ctr.set_lookat([0.0, 0.0, 0.5])
                ctr.set_front([0.0, 0.0, 1.0])
                ctr.set_up([0.0, 1.0, 0.0])
                ctr.set_zoom(0.35)
            else:
                # Use generator camera pose (translation from trajectory.json, rotation from poses_tum.txt if present)
                idx = self.current_idx
                front = np.array([0.0, 0.0, 1.0], dtype=float)
                up = np.array([0.0, 1.0, 0.0], dtype=float)
                pos = np.zeros(3, dtype=float)
                if idx < len(self.camera_positions) and self.camera_positions[idx] is not None:
                    pos = np.asarray(self.camera_positions[idx], dtype=float)
                if hasattr(self, 'tum_poses') and self.tum_poses and idx < len(self.tum_poses):
                    qx, qy, qz, qw = self.tum_poses[idx]['q']
                    x, y, z, w = qx, qy, qz, qw
                    R = np.array([
                        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
                        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
                    ], dtype=float)
                    front = R @ np.array([0.0, 0.0, 1.0], dtype=float)
                    up = R @ np.array([0.0, 1.0, 0.0], dtype=float)
                front = front / (np.linalg.norm(front) + 1e-9)
                up = up / (np.linalg.norm(up) + 1e-9)
                lookat = (pos + 0.5 * front).tolist()
                ctr.set_lookat(lookat)
                ctr.set_front(front.tolist())
                ctr.set_up(up.tolist())
                ctr.set_zoom(0.35)
        else:
            ctr.set_lookat([0, 0, 0])
            ctr.set_front([0, -1, 0])
            ctr.set_up([0, 0, 1])
            ctr.set_zoom(0.35)

    # Single-frame viewer

    def view_single_frame(self, fixed_frame_num=None):
        total = len(self.frame_files)
        if fixed_frame_num is None:
            print(f"Available frames: 1..{total}")
            try:
                chosen = int(input(f"Enter frame number (1..{total}): "))
            except Exception:
                chosen = 1
            chosen = min(max(1, chosen), total)
        else:
            chosen = min(max(1, fixed_frame_num), total)

        self.current_idx = chosen - 1
        frame_data = self.load_frame(self.current_idx)
        if frame_data is None:
            print("Failed to load frame.")
            return

        geoms = self.create_geometries(frame_data)

        # Open3D window
        self._vis = o3d.visualization.VisualizerWithKeyCallback()
        self._vis.create_window(window_name=f"LiDAR Viewer - Frame {frame_data['frame']}", width=1280, height=900)

        # Add point cloud first, then camera elements to ensure camera is not occluded
        self._pcd = geoms['pcd']
        self._vis.add_geometry(self._pcd)

        # Retain camera elements for callbacks and ensure they're rendered on top
        self._cam_sphere = geoms['cam_sphere']
        self._coord_frame = geoms['coord_frame']
        self._line_set = geoms.get('line_set')

        # Ensuring camera sphere and axes are added after point cloud for better visibility
        self._vis.add_geometry(self._cam_sphere)
        self._vis.add_geometry(self._coord_frame)
        if self._line_set is not None:
            self._vis.add_geometry(self._line_set)

        # Info with enhanced view mode description
        print(f"\n📋 Frame {frame_data['frame']}  |  Points: {len(frame_data['points']):,}")
        print(f"🎨 Coloring: {geoms['color_desc']}")
        print(f"👁️ View: {self.view_mode} mode" + (" (LiDAR POV)" if self.view_mode == 'camera' else " (External view)"))
        print(f"📡 Camera position: {frame_data['camera_location']}")
        print(f"⌨️  Controls: R=cycle color, T=toggle trajectory, V=toggle view (world/camera POV)")

        # Key callbacks
        self._vis.register_key_callback(ord('R'), lambda vis: self._on_cycle_color(frame_data))
        self._vis.register_key_callback(ord('T'), lambda vis: self._on_toggle_trajectory(frame_data))
        self._vis.register_key_callback(ord('V'), lambda vis: self._on_toggle_view())

        self._apply_view_mode()
        self._vis.run()
        self._vis.destroy_window()

    # Callbacks

    def _on_cycle_color(self, frame_data):
        i = self.COLOR_MODES.index(self.color_mode)
        self.color_mode = self.COLOR_MODES[(i + 1) % len(self.COLOR_MODES)]
        colors, desc = self.color_by_mode(frame_data, self.color_mode)
        self._pcd.colors = o3d.utility.Vector3dVector(colors)
        self._vis.update_geometry(self._pcd)
        self._vis.update_renderer()
        print(f"Color mode → {self.color_mode} ({desc})")
        return True

    def _on_toggle_trajectory(self, frame_data):
        self.show_trajectory = not self.show_trajectory
        if self._line_set is not None:
            self._vis.remove_geometry(self._line_set, reset_bounding_box=False)
            self._line_set = None
        if self.show_trajectory:
            geoms = self.create_geometries(frame_data)
            if 'line_set' in geoms:
                self._line_set = geoms['line_set']
                self._vis.add_geometry(self._line_set)
        self._vis.update_renderer()
        print(f"Trajectory → {'ON' if self.show_trajectory else 'OFF'}")
        return True

    def _on_toggle_view(self):
        self.view_mode = 'camera' if self.view_mode == 'world' else 'world'
        self._apply_view_mode()
        self._vis.update_renderer()
        print(f"View mode → {self.view_mode}")
        return True


# CLI

def main():
    parser = argparse.ArgumentParser(description="View LiDAR point clouds from Infinigen's generator")
    parser.add_argument('data_dir', help='Directory with lidar_frame_*.ply and metadata JSONs')
    parser.add_argument('--color', choices=LidarViewer.COLOR_MODES, default='intensity',
                        help='Initial point coloring mode')
    parser.add_argument('--frame', type=int, default=None, help='Frame number to view (1-based). If omitted, prompts.')
    parser.add_argument('--view', choices=['world', 'camera'], default='world', 
                        help="Initial view mode: 'camera' for LiDAR POV, 'world' for external view")
    parser.add_argument('--camera-view', action='store_true', 
                        help="Shortcut to use camera POV view (same as --view camera)")
    parser.add_argument('--no-trajectory', action='store_true', help='Disable trajectory overlay')

    args = parser.parse_args()

    if not _HAVE_PLYFILE:
        print("Note: 'plyfile' not installed → auxiliary PLY fields may be unavailable. Install: pip install plyfile")

    viewer = LidarViewer(args.data_dir, color_mode=args.color)
    # Set view mode (camera-view option takes precedence if specified)
    viewer.view_mode = 'camera' if args.camera_view else args.view
    viewer.show_trajectory = not args.no_trajectory
    viewer.view_single_frame(fixed_frame_num=args.frame)


if __name__ == '__main__':
    main()
