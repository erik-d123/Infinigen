# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Lightweight viewer for LiDAR PLY outputs.

Loads point clouds with optional attributes and displays them in Open3D with
simple color modes (intensity, reflectivity, ring). Includes a tolerant ASCII
PLY reader for environments without `plyfile`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    from plyfile import PlyData  # richer PLY reader for custom fields
except Exception:
    PlyData = None


def _list_frames(out_dir: Path) -> List[Path]:
    """Return sorted PLY paths in a directory, raising if empty."""
    files = sorted(out_dir.glob("lidar_frame_*.ply"))
    if not files:
        raise FileNotFoundError(f"No PLYs found in {out_dir}")
    return files


def _read_ply_ascii_tolerant(path: Path) -> dict:
    """Fallback ASCII PLY reader.

    Coerces integer fields that may appear as float literals (e.g. "0.0" → 0).
    """

    cols = {}
    props = []
    with open(path, "r", encoding="utf-8") as f:
        in_header = True
        for line in f:
            line = line.strip()
            if in_header:
                if line.startswith("property"):
                    parts = line.split()
                    if len(parts) >= 3:
                        # e.g., 'property uchar mat_class' -> (name, dtype)
                        props.append((parts[2], parts[1].lower()))
                elif line == "end_header":
                    in_header = False
                    cols = {name: [] for name, _ in props}
                continue
            if not line:
                continue
            toks = line.split()
            if len(toks) < len(props):
                continue
            for j, (name, dtype) in enumerate(props):
                tok = toks[j]
                if dtype in {
                    "uchar",
                    "char",
                    "ushort",
                    "short",
                    "uint",
                    "int",
                    "uint8",
                    "uint16",
                }:
                    try:
                        v = int(tok)
                    except ValueError:
                        v = int(float(tok))
                else:
                    v = float(tok)
                cols[name].append(v)
    out = {}

    if not cols:
        return out

    def arr(name):
        return np.asarray(cols[name]) if name in cols else None

    x, y, z = arr("x"), arr("y"), arr("z")
    if x is not None and y is not None and z is not None:
        out["points"] = np.stack([x, y, z], axis=1).astype("f4")

    def put(name, dt):
        a = arr(name)
        if a is not None:
            out[name] = a.astype(dt, copy=False)

    for name, dt in [
        ("intensity", "u1"),
        ("ring", "u2"),
        ("azimuth", "f4"),
        ("elevation", "f4"),
        ("return_id", "u1"),
        ("num_returns", "u1"),
        ("range_m", "f4"),
        ("cos_incidence", "f4"),
        ("mat_class", "u1"),
        ("reflectivity", "f4"),
        ("transmittance", "f4"),
    ]:
        put(name, dt)
    if all(k in cols for k in ("nx", "ny", "nz")):
        out["normals"] = np.stack([arr("nx"), arr("ny"), arr("nz")], axis=1).astype(
            "f4"
        )
    return out


def _read_ply_all(path: Path) -> Dict[str, np.ndarray]:
    """Read xyz and optional attributes; prefer plyfile, fallback to Open3D."""
    data: Dict[str, np.ndarray] = {}
    if PlyData is not None:
        try:
            pd = PlyData.read(str(path))
            v = pd["vertex"]
        except Exception:
            return _read_ply_ascii_tolerant(path)

        def col(name, dtype=None):
            if name in v.data.dtype.names:
                arr = np.asarray(v.data[name])
                return arr.astype(dtype) if dtype is not None else arr
            return None

        x = col("x", "f4")
        y = col("y", "f4")
        z = col("z", "f4")
        pts = np.stack([x, y, z], axis=1).astype("f4")
        data["points"] = pts
        for k, dt in [
            ("intensity", "u1"),
            ("ring", "u2"),
            ("azimuth", "f4"),
            ("elevation", "f4"),
            ("return_id", "u1"),
            ("num_returns", "u1"),
            ("range_m", "f4"),
            ("cos_incidence", "f4"),
            ("mat_class", "u1"),
            ("reflectivity", "f4"),
            ("transmittance", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
        ]:
            arr = col(k, dt)
            if arr is not None:
                data[k] = arr
        # normals (nx,ny,nz) if present
        if all(k in data for k in ("nx", "ny", "nz")):
            data["normals"] = np.stack(
                [data.pop("nx"), data.pop("ny"), data.pop("nz")], axis=1
            ).astype("f4")
        return data

    # Fallback to Open3D geometry only
    assert o3d is not None, "open3d required if plyfile unavailable"
    pcd = o3d.io.read_point_cloud(str(path))
    data["points"] = np.asarray(pcd.points, dtype="f4")
    return data


def _color_from_intensity_u8(intensity: np.ndarray, N: int) -> np.ndarray:
    """Compute grayscale colors from 8‑bit intensity with robust scaling."""
    if intensity is None or intensity.size != N:
        return np.zeros((N, 3), dtype="f4")
    a = intensity.astype(np.float32)
    # Stretch 5th..99th percentiles -> [0,1]
    lo, hi = np.percentile(a, 5), np.percentile(a, 99)
    if not (hi > lo):  # degenerate
        g = (a / 255.0).clip(0, 1)
    else:
        g = ((a - lo) / (hi - lo)).clip(0, 1)
    return np.stack([g, g, g], axis=1)


def _color_from_reflectivity(refl: np.ndarray, N: int) -> np.ndarray:
    """Compute grayscale colors from [0,1] reflectivity."""
    if refl is None or refl.size != N:
        return np.zeros((N, 3), dtype="f4")
    g = np.asarray(refl, dtype=np.float32).clip(0.0, 1.0)
    return np.stack([g, g, g], axis=1)


def _tab20(n: int) -> np.ndarray:
    """A small repeating qualitative palette (TAB20‑like)."""
    base = (
        np.array(
            [
                [31, 119, 180],
                [255, 127, 14],
                [44, 160, 44],
                [214, 39, 40],
                [148, 103, 189],
                [140, 86, 75],
                [227, 119, 194],
                [127, 127, 127],
                [188, 189, 34],
                [23, 190, 207],
                [174, 199, 232],
                [255, 187, 120],
                [152, 223, 138],
                [255, 152, 150],
                [197, 176, 213],
                [196, 156, 148],
                [247, 182, 210],
                [199, 199, 199],
                [219, 219, 141],
                [158, 218, 229],
            ],
            dtype=np.float32,
        )
        / 255.0
    )
    return np.vstack([base for _ in range((n + 19) // 20)])[:n]


def _color_from_ring(ring: np.ndarray, N: int) -> np.ndarray:
    """Colorize points by ring index using a repeating palette."""
    if ring is None or ring.size != N:
        return np.zeros((N, 3), dtype="f4")
    r = ring.astype(np.int64)
    unique = np.unique(r)
    lut = _tab20(unique.size)
    idx = {v: i for i, v in enumerate(unique.tolist())}
    c = np.asarray([lut[idx[int(v)]] for v in r], dtype="f4")
    return c


def _intensity_to_heat(intensity: np.ndarray, N: int) -> np.ndarray:
    """Pseudo‑color intensity to a simple JET‑like heatmap."""
    if intensity is None or intensity.size != N:
        return np.zeros((N, 3), dtype="f4")
    a = intensity.astype(np.float32)
    lo, hi = np.percentile(a, 5), np.percentile(a, 99)
    if not (hi > lo):
        g = (a / 255.0).clip(0, 1)
    else:
        g = ((a - lo) / (hi - lo)).clip(0, 1)

    # crude JET: blue->cyan->green->yellow->red
    def lerp(x, a, b):
        return a + (b - a) * x

    R = np.where(
        g < 0.5, 0.0, np.where(g < 0.75, lerp((g - 0.5) / 0.25, 0.0, 1.0), 1.0)
    )
    G = np.where(
        g < 0.25,
        lerp(g / 0.25, 0.0, 1.0),
        np.where(g < 0.75, 1.0, lerp((g - 0.75) / 0.25, 1.0, 0.0)),
    )
    B = np.where(
        g < 0.25, 1.0, np.where(g < 0.5, lerp((g - 0.25) / 0.25, 1.0, 0.0), 0.0)
    )
    return np.stack([R, G, B], axis=1).astype("f4")


def _make_o3d_pcd(
    points: np.ndarray, colors: np.ndarray, normals: Optional[np.ndarray] = None
):
    """Create an Open3D point cloud with optional normals."""
    assert o3d is not None, "open3d is required for visualization"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None and normals.shape == points.shape:
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return pcd


def _apply_color_mode(data: Dict[str, np.ndarray], mode: str) -> np.ndarray:
    """Select a color mapping for the viewer from available fields."""
    pts = data["points"]
    N = pts.shape[0]
    if mode == "reflectivity" and "reflectivity" in data:
        return _color_from_reflectivity(data["reflectivity"], N)
    if mode == "ring" and "ring" in data:
        return _color_from_ring(data["ring"], N)
    if mode == "intensity_heat" and "intensity" in data:
        return _intensity_to_heat(data.get("intensity", None), N)
    return _color_from_intensity_u8(data.get("intensity", None), N)


def _load_frame(
    path: Path, mode: str
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
    """Load a single frame, returning (points, colors, normals, aux)."""
    data = _read_ply_all(path)
    pts = data["points"]
    cols = _apply_color_mode(data, mode)
    norms = data.get("normals")
    return pts, cols, norms, data


def _title(dir_path: Path, files: List[Path], idx: int, mode: str) -> str:
    """Format a window title for the current frame and color mode."""
    name = files[idx].name
    return f"{dir_path.name}  |  {name}  |  color: {mode}"


def view_dir(out_dir: str, color: str = "intensity", frame: Optional[int] = None):
    """Open an interactive viewer for a directory of LiDAR frames."""
    assert o3d is not None, "open3d is required for visualization"
    outp = Path(out_dir)
    files = _list_frames(outp)

    # initial index
    if frame is not None:
        # frame number is inferred from filename suffix
        fname = f"lidar_frame_{int(frame):04d}.ply"
        try:
            idx = files.index(outp / fname)
        except ValueError:
            idx = 0
    else:
        idx = 0

    mode = (
        color
        if color in ("intensity", "intensity_heat", "reflectivity", "ring")
        else "intensity"
    )
    pts, cols, norms, aux = _load_frame(files[idx], mode)
    pcd = _make_o3d_pcd(pts, cols, norms)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=_title(outp, files, idx, mode), width=1280, height=800
    )
    vis.add_geometry(pcd)

    def reload_geometry(new_idx: int, new_mode: str):
        nonlocal idx, mode, pcd
        idx = new_idx % len(files)
        mode = new_mode
        pts, cols, norms, _ = _load_frame(files[idx], mode)
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        if norms is not None and norms.shape == pts.shape:
            pcd.normals = o3d.utility.Vector3dVector(norms.astype(np.float64))
        vis.update_geometry(pcd)
        vis.get_render_option().point_size = 2.0
        vis.update_renderer()
        vis.reset_view_point(True)
        vis.get_window_name()

    # Key callbacks
    vis.register_key_callback(ord("N"), lambda v: reload_geometry(idx + 1, mode))
    modes = ["intensity", "intensity_heat", "reflectivity", "ring"]
    vis.register_key_callback(
        ord("C"),
        lambda v: reload_geometry(idx, modes[(modes.index(mode) + 1) % len(modes)]),
    )
    vis.register_key_callback(ord("Q"), lambda v: vis.close())

    vis.run()
    vis.destroy_window()


def parse_args(argv=None):
    """CLI parser for the LiDAR viewer."""
    ap = argparse.ArgumentParser("Infinigen indoor LiDAR viewer")
    ap.add_argument("output_dir", type=str, help="Directory with lidar_frame_*.ply")
    ap.add_argument(
        "--color",
        type=str,
        default="intensity",
        choices=["intensity", "intensity_heat", "reflectivity", "ring"],
    )
    ap.add_argument("--frame", type=int, default=None)
    return ap.parse_args(argv)


def main(argv=None):
    """Entry point for the LiDAR viewer."""
    args = parse_args(argv)
    view_dir(args.output_dir, color=args.color, frame=args.frame)


if __name__ == "__main__":
    main()
