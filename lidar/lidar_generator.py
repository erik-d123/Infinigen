#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import sys
import time
import numpy as np
import bpy
from mathutils import Matrix

# Local modules - robust import strategy that works in all contexts
import os
import sys

# Add the current script's directory to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

# First try relative import (when imported as part of a package)
try:
    from .lidar_config import LidarConfig, LIDAR_PRESETS
    from .lidar_scene import setup_scene, resolve_camera, sensor_to_camera_rotation
    from .lidar_raycast import generate_sensor_rays, perform_raycasting
    from .lidar_io import save_ply, world_to_frame_matrix, _matrix_to_np3x3
except (ImportError, ValueError):
    # Fall back to absolute import (when run as a script)
    from lidar_config import LidarConfig, LIDAR_PRESETS
    from lidar_scene import setup_scene, resolve_camera, sensor_to_camera_rotation
    from lidar_raycast import generate_sensor_rays, perform_raycasting
    from lidar_io import save_ply, world_to_frame_matrix, _matrix_to_np3x3

def process_frame(scene, cam_obj, frame, fps, output_dir, cfg: LidarConfig, sensor_R: Matrix, precomp, phase_offset_rad,
                  *, write_ply=True):
    # Process single frame: generate rays, cast, and save results
    bpy.context.scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()

    dirs_sensor, ring_ids, az_idx, elev_rad, az_base = precomp

    # Timing calculations for sensor spin
    frame_dt = 1.0 / fps
    rps = cfg.rpm / 60.0
    omega = 2.0 * math.pi * rps  # rad/s
    col_frac = az_idx.astype(np.float32) / float(cfg.num_azimuth)

    if cfg.continuous_spin:
        if cfg.rolling_shutter:
            rev_time = 1.0 / max(rps, 1e-6)
            t_offsets = (col_frac * rev_time).astype(np.float32)
        else:
            t_offsets = np.zeros_like(col_frac, dtype=np.float32)
        az = az_base + phase_offset_rad
    else:
        t_offsets = np.zeros_like(col_frac, dtype=np.float32)
        az = az_base + phase_offset_rad

    # Apply azimuth rotation to sensor ray directions
    ca, sa = np.cos(az), np.sin(az)
    dirs_yawed = np.stack([
        dirs_sensor[:, 0] * ca - dirs_sensor[:, 1] * sa,
        dirs_sensor[:, 0] * sa + dirs_sensor[:, 1] * ca,
        dirs_sensor[:, 2]
    ], axis=1)

    # Transform sensor directions to world coordinates
    R_cam_np = _matrix_to_np3x3(cam_obj.matrix_world.to_3x3())
    sensor_to_cam_np = _matrix_to_np3x3(sensor_R)
    world_dirs_np = dirs_yawed @ sensor_to_cam_np.T @ R_cam_np.T
    world_dirs = world_dirs_np / np.linalg.norm(world_dirs_np, axis=1, keepdims=True)

    origin = cam_obj.matrix_world.translation

    # Perform ray casting
    t0 = time.time()
    res = perform_raycasting(scene, depsgraph, origin, world_dirs, ring_ids, az, elev_rad, t_offsets, cfg)
    dt = time.time() - t0

    if not res:
        print(f"Frame {frame}: cast {len(world_dirs)} rays, hit 0 points ({dt:.2f}s)")
        return 0, phase_offset_rad, None

    nhit = len(res["points_world"])
    print(f"Frame {frame}: cast {len(world_dirs)} rays, hit {nhit} points ({dt:.2f}s, scale={res['scale_used']:.5f})")

    # Save output files
    xform = world_to_frame_matrix(cam_obj, sensor_R, cfg.ply_frame)
    if write_ply and cfg.save_ply:
        save_ply(output_dir, frame, xform, res)

    # Update phase for continuous spin
    if cfg.continuous_spin:
        phase_offset_rad = (phase_offset_rad + omega * frame_dt) % (2.0 * math.pi)

    return nhit, phase_offset_rad, res

def parse_args(argv):
    # Focused, minimal CLI for indoor LiDAR GT
    p = argparse.ArgumentParser(description="Generate indoor LiDAR point clouds (raycast).")
    p.add_argument("scene_path", help="Path to Blender .blend scene")
    p.add_argument("--output_dir", default="outputs/infinigen_lidar", help="Output directory")
    p.add_argument("--frames", default="1-48", help="Frame range, e.g. '1-48' or '1,5,10'")
    p.add_argument("--camera", default=None, help="Camera object name")
    p.add_argument("--preset", default="VLP-16", choices=LIDAR_PRESETS.keys(), help="LiDAR sensor preset")
    p.add_argument("--force-azimuth-steps", type=int, default=None, help="Override azimuth columns")
    p.add_argument("--ply-frame", choices=["sensor", "camera", "world"], default="sensor", help="PLY frame")
    p.add_argument("--secondary", action="store_true", help="Enable pass-through secondary returns")
    p.add_argument("--secondary-min-residual", type=float, default=0.05, help="Minimum residual transmission to spawn a secondary return")
    p.add_argument("--secondary-ray-bias", type=float, default=5e-4, help="Offset (m) applied when spawning secondary rays past transmissive surfaces")
    p.add_argument("--secondary-extinction", type=float, default=0.0, help="Extinction coefficient (1/m) for transmissive media")
    p.add_argument("--specular-falloff-power", type=float, default=1.0, help="Exponent for specular intensity falloff with incidence angle")
    p.add_argument("--auto-expose", action="store_true", help="Enable per-frame percentile exposure scaling for U8 intensity")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    return p.parse_args(argv)

def parse_frame_list(spec: str):
    # Parse frame specification into list of frame numbers
    if "-" in spec:
        a, b = map(int, spec.split("-"))
        return list(range(a, b + 1))
    return list(map(int, spec.split(",")))

def main():
    # Main execution function
    script_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parse_args(script_args)

    # Parse inputs and setup scene
    frames = parse_frame_list(args.frames)
    scene = setup_scene(args.scene_path)
    cam = resolve_camera(args.camera)

    # Seed RNGs if requested
    if args.seed is not None:
        try:
            import random
            random.seed(args.seed)
            np.random.seed(args.seed % (2**32 - 1))
        except Exception:
            pass

    # Create configuration
    cfg = LidarConfig(
        preset=args.preset,
        force_azimuth_steps=args.force_azimuth_steps,
        save_ply=True,
        global_scale=1.0,
        rpm=None,
        continuous_spin=True,
        rolling_shutter=True,
        ply_frame=args.ply_frame,
        auto_expose=args.auto_expose,
        enable_secondary=args.secondary,
        secondary_min_residual=args.secondary_min_residual,
        secondary_ray_bias=args.secondary_ray_bias,
        specular_falloff_power=args.specular_falloff_power,
        secondary_extinction=args.secondary_extinction,
    )

    # Setup output directory and save configuration
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "lidar_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    # Prepare sensor configuration and ray patterns
    sensor_R = sensor_to_camera_rotation()
    precomp = generate_sensor_rays(cfg)

    # Initialize processing state
    total_pts = 0
    phase = 0.0
    metadata = {"frames": {}}
    traj = {}

    fps = scene.render.fps / max(scene.render.fps_base, 1.0)
    t_all = time.time()

    print(f"Processing {len(frames)} frames with {cfg.preset} sensor...")
    print(f"Configuration: {cfg}")

    # Process each frame
    for fr in frames:
        nhit, phase, res = process_frame(
            scene, cam, fr, fps, args.output_dir, cfg, sensor_R, precomp, phase,
            write_ply=True,
        )
        total_pts += nhit

        # Store camera trajectory
        Mw = cam.matrix_world
        t = Mw.translation
        traj[fr] = {"t": [t[0], t[1], t[2]]}

        metadata["frames"][fr] = {
            "points": int(nhit), 
            "scale_used": float(res["scale_used"]) if res else 1.0
        }

    # Save trajectory and metadata
    with open(os.path.join(args.output_dir, "trajectory.json"), "w", encoding="utf-8") as f:
        json.dump(traj, f, indent=2)
    with open(os.path.join(args.output_dir, "frame_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Export timestamps (derived from FPS) and TUM-style poses
    t0 = frames[0]
    timestamps = [(fr - t0) / fps for fr in frames]
    with open(os.path.join(args.output_dir, "timestamps.txt"), "w", encoding="utf-8") as f:
        for ts in timestamps:
            f.write(f"{ts:.6f}\n")
    with open(os.path.join(args.output_dir, "poses_tum.txt"), "w", encoding="utf-8") as f:
        for fr, ts in zip(frames, timestamps):
            bpy.context.scene.frame_set(fr)
            Mw = cam.matrix_world
            loc = Mw.translation
            quat = Mw.to_quaternion()  # (w, x, y, z)
            # TUM: timestamp tx ty tz qx qy qz qw
            f.write(
                f"{ts:.6f} {loc[0]:.9f} {loc[1]:.9f} {loc[2]:.9f} {quat.x:.9f} {quat.y:.9f} {quat.z:.9f} {quat.w:.9f}\n"
            )


# Convenience entry point for running on the currently-loaded Blender scene.
# Useful when integrating as a pipeline stage without re-opening the .blend.
def run_on_current_scene(output_dir: str,
                         frames: list[int],
                         camera_name: str | None,
                         cfg: LidarConfig):
    scene = bpy.context.scene
    cam = resolve_camera(camera_name)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "lidar_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    sensor_R = sensor_to_camera_rotation()
    precomp = generate_sensor_rays(cfg)
    fps = scene.render.fps / max(scene.render.fps_base, 1.0)

    total_pts = 0
    phase = 0.0
    metadata = {"frames": {}}
    traj = {}

    for fr in frames:
        nhit, phase, res = process_frame(
            scene, cam, fr, fps, output_dir, cfg, sensor_R, precomp, phase,
            write_ply=True, write_kitti=False,
        )
        total_pts += nhit

        Mw = cam.matrix_world
        t = Mw.translation
        traj[fr] = {"t": [t[0], t[1], t[2]]}

        metadata["frames"][fr] = {
            "points": int(nhit),
            "scale_used": float(res["scale_used"]) if res else 1.0
        }

    with open(os.path.join(output_dir, "trajectory.json"), "w", encoding="utf-8") as f:
        json.dump(traj, f, indent=2)
    with open(os.path.join(output_dir, "frame_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # timestamps and TUM poses for in-memory run
    t0 = frames[0]
    timestamps = [(fr - t0) / fps for fr in frames]
    with open(os.path.join(output_dir, "timestamps.txt"), "w", encoding="utf-8") as f:
        for ts in timestamps:
            f.write(f"{ts:.6f}\n")
    with open(os.path.join(output_dir, "poses_tum.txt"), "w", encoding="utf-8") as f:
        for fr, ts in zip(frames, timestamps):
            bpy.context.scene.frame_set(fr)
            Mw = cam.matrix_world
            loc = Mw.translation
            quat = Mw.to_quaternion()
            f.write(
                f"{ts:.6f} {loc[0]:.9f} {loc[1]:.9f} {loc[2]:.9f} {quat.x:.9f} {quat.y:.9f} {quat.z:.9f} {quat.w:.9f}\n"
            )

    print(f"Frames: {len(frames)}, Total points: {total_pts:,}")
    print(f"Output: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()
