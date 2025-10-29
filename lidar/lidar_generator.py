# SPDX-License-Identifier: MIT
# Orchestration for indoor LiDAR: CLI + per-frame generation

from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

try:
    import bpy
except Exception:
    bpy = None

from lidar.lidar_config import LidarConfig
from lidar.lidar_raycast import generate_sensor_rays, perform_raycasting
from lidar.lidar_scene import resolve_camera, sensor_to_camera_rotation
from lidar.lidar_io import save_ply

# ------------------------------- utils -------------------------------

def _parse_frames(frames_arg: str) -> List[int]:
    """Accept '1-48', '1,5,10', or single '12'."""
    s = str(frames_arg).strip()
    if "-" in s:
        a, b = s.split("-", 1)
        a = int(a); b = int(b)
        lo, hi = (a, b) if a <= b else (b, a)
        return list(range(lo, hi + 1))
    if "," in s:
        return sorted({int(x) for x in s.split(",") if x})
    return [int(s)]

def _frame_time_seconds(scene, frame: int) -> float:
    fps = scene.render.fps / max(1, scene.render.fps_base)
    return (frame - 1) / max(1.0, float(fps))

def _sensor_world_rotation(camera_obj) -> np.ndarray:
    """R_world_sensor: world←sensor."""
    R_cam_world = np.array(camera_obj.matrix_world.to_3x3(), dtype=float)
    R_cam_sensor = np.array(sensor_to_camera_rotation(), dtype=float)  # camera←sensor
    return R_cam_world @ R_cam_sensor

def _world_from_sensor(camera_obj, dirs_sensor: np.ndarray) -> np.ndarray:
    """Rotate sensor-frame rays to world."""
    R = _sensor_world_rotation(camera_obj)  # world←sensor
    return (R @ dirs_sensor.reshape(-1, 3).T).T.reshape(dirs_sensor.shape)

def _origins_for(camera_obj, N: int) -> np.ndarray:
    loc = np.array(camera_obj.matrix_world.translation, dtype=np.float64)
    return np.repeat(loc[None, :], N, axis=0)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# --------------------------- frame processing ---------------------------

@dataclass
class FrameOutputs:
    ply_path: Path
    scale_used: float
    points: int

def process_frame(scene, camera_obj, cfg: LidarConfig, out_dir: Path, frame: int) -> FrameOutputs:
    """Emit LiDAR for a single frame and save PLY + metadata JSON lines."""
    assert bpy is not None, "Must run inside Blender"

    # Set frame and update depsgraph
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()

    # Rays in sensor frame
    rays = generate_sensor_rays(cfg)
    dirs_sensor = rays["directions"]  # (R,A,3), +X forward sensor frame
    rings = np.repeat(np.arange(dirs_sensor.shape[0], dtype=np.int16), dirs_sensor.shape[1])
    az = np.tile(np.linspace(-math.pi, math.pi, dirs_sensor.shape[1], endpoint=False).astype(np.float32),
                 dirs_sensor.shape[0])

    # World transform
    dirs_world = _world_from_sensor(camera_obj, dirs_sensor).reshape(-1, 3).astype(np.float64)
    dirs_world /= np.linalg.norm(dirs_world, axis=1, keepdims=True).clip(1e-12, None)
    origins = _origins_for(camera_obj, dirs_world.shape[0])

    # Cast
    res = perform_raycasting(
        scene=scene,
        depsgraph=deps,
        origins=origins,
        directions=dirs_world,
        rings=rings.astype(np.uint16),
        azimuth_rad=az,
        cfg=cfg,
    )
    # Frame transform for PLY
    # sensor: +X fwd, +Y left, +Z up; camera: Blender camera; world: as-is
    frame_mode = getattr(cfg, "ply_frame", "sensor")
    pts_world = res["points"]
    if frame_mode == "world":
        pts_out = pts_world
    else:
        # world→camera
        cam = camera_obj
        R_wc = np.array(cam.matrix_world.to_3x3(), dtype=float)
        t_wc = np.array(cam.matrix_world.translation, dtype=float)
        R_cw = R_wc.T
        pts_cam = (R_cw @ (pts_world - t_wc).T).T
        if frame_mode == "camera":
            pts_out = pts_cam
        elif frame_mode == "sensor":
            # camera→sensor
            R_cs = np.array(sensor_to_camera_rotation(), dtype=float)  # camera←sensor
            R_sc = R_cs.T
            pts_out = (R_sc @ pts_cam.T).T
        else:
            pts_out = pts_world  # fallback

    # Save PLY
    out_ply = out_dir / f"lidar_frame_{frame:04d}.ply"
    save_ply(
        out_ply,
        {
            "points": pts_out.astype(np.float32),
            "intensity": res["intensity"],
            "ring": res["ring"],
            "azimuth": res["azimuth"],
            "elevation": res["elevation"],
            "return_id": res["return_id"],
            "num_returns": res["num_returns"],
            "range_m": res.get("range_m"),
            "cos_incidence": res.get("cos_incidence"),
            "mat_class": res.get("mat_class"),
            "reflectivity": res.get("reflectivity"),
            "transmittance": res.get("transmittance"),
        },
        binary=getattr(cfg, "ply_binary", False),
    )

    # Per-frame metadata
    meta = {"frame": frame, "points": int(pts_out.shape[0]), "scale_used": float(res.get("scale_used", 0.0))}
    (out_dir / "frame_metadata.jsonl").write_text(
        ((out_dir / "frame_metadata.jsonl").read_text() if (out_dir / "frame_metadata.jsonl").exists() else "")
        + json.dumps(meta) + "\n"
    )

    return FrameOutputs(out_ply, float(res.get("scale_used", 0.0)), int(pts_out.shape[0]))

# ------------------------- top-level entrypoints -------------------------

def run_on_current_scene(output_dir: str, frames: Sequence[int], camera_name: str, cfg_kwargs=None):
    """Operate on the current open Blender scene."""
    assert bpy is not None, "Must run inside Blender"
    cfg_kwargs = dict(cfg_kwargs or {})
    cfg = LidarConfig(**cfg_kwargs)

    scene = bpy.context.scene
    cam = resolve_camera(scene, camera_name)
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)

    # Save config used
    (out_dir / "lidar_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    # Trajectory container
    traj = {"frames": {}, "t": []}

    # Seed
    seed = int(cfg_kwargs.get("seed", 0)) if "seed" in cfg_kwargs else None
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    for f in frames:
        fo = process_frame(scene, cam, cfg, out_dir, f)
        # Minimal trajectory (translation only)
        t = list(map(float, cam.matrix_world.translation))
        traj["frames"][str(f)] = {"points": fo.points, "scale_used": fo.scale_used}
        traj["t"].append({"frame": f, "t": t})

    # Write trajectory and timestamps
    (out_dir / "trajectory.json").write_text(json.dumps(traj, indent=2))
    ts_path = out_dir / "timestamps.txt"
    with ts_path.open("w") as fh:
        for f in frames:
            fh.write(f"{_frame_time_seconds(scene, f):.9f}\n")

def generate_for_scene(scene_path: str, output_dir: str, frames: Sequence[int], camera_name="Camera", cfg_kwargs=None):
    """Open a .blend and run LiDAR generation."""
    assert bpy is not None, "Must run inside Blender"
    bpy.ops.wm.open_mainfile(filepath=str(scene_path))
    run_on_current_scene(output_dir=output_dir, frames=frames, camera_name=camera_name, cfg_kwargs=cfg_kwargs)

# -------------------------------- CLI --------------------------------

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser("Infinigen indoor LiDAR")
    p.add_argument("scene_path", type=str, help="Path to .blend")
    p.add_argument("--output_dir", type=str, default="outputs/infinigen_lidar", help="Output directory")
    p.add_argument("--frames", type=str, default="1", help="e.g. '1-48' or '1,5,10'")
    p.add_argument("--camera", type=str, default="Camera", help="Camera object name")
    p.add_argument("--preset", type=str, default="VLP-16")
    p.add_argument("--force-azimuth-steps", type=int, default=None)
    p.add_argument("--ply-frame", type=str, default="sensor", choices=["sensor", "camera", "world"])
    p.add_argument("--secondary", action="store_true")
    p.add_argument("--secondary-min-residual", type=float, default=0.05)
    p.add_argument("--secondary-ray-bias", type=float, default=5e-4)
    p.add_argument("--secondary-min-cos", type=float, default=0.95)
    p.add_argument("--secondary-merge-eps", type=float, default=0.0)
    p.add_argument("--auto-expose", action="store_true")
    p.add_argument("--global-scale", type=float, default=1.0)
    p.add_argument("--default-opacity", type=float, default=1.0)
    p.add_argument("--ply-binary", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)

def main(argv: Sequence[str] | None = None):
    args = parse_args(sys.argv[sys.argv.index("--") + 1:] if argv is None and "--" in sys.argv else (argv or sys.argv[1:]))

    cfg_kwargs = dict(
        preset=args.preset,
        force_azimuth_steps=args.force_azimuth_steps,
        ply_frame=args.ply_frame,
        enable_secondary=bool(args.secondary),
        secondary_min_residual=args.secondary_min_residual,
        secondary_ray_bias=args.secondary_ray_bias,
        secondary_min_cos=args.secondary_min_cos,
        secondary_merge_eps=args.secondary_merge_eps,
        auto_expose=bool(args.auto_expose),
        global_scale=args.global_scale,
        default_opacity=args.default_opacity,
        ply_binary=bool(args.ply_binary),
        prefer_ior=True,
    )

    frames = _parse_frames(args.frames)
    generate_for_scene(
        scene_path=args.scene_path,
        output_dir=args.output_dir,
        frames=frames,
        camera_name=args.camera,
        cfg_kwargs=cfg_kwargs,
    )

if __name__ == "__main__":
    main()