from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import bpy
import numpy as np

from infinigen.core.util import camera as util_cam
from lidar.lidar_config import LidarConfig
from lidar.lidar_io import save_ply
from lidar.lidar_raycast import generate_sensor_rays, perform_raycasting
from lidar.lidar_scene import resolve_camera, sensor_to_camera_rotation


def _parse_frames(frames_arg: str) -> List[int]:
    """Accept '1-48', '1,5,10', or single '12'."""
    s = str(frames_arg).strip()
    if "-" in s:
        a, b = s.split("-", 1)
        a = int(a)
        b = int(b)
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


def _get_cam_ids(camera_obj) -> tuple[int, int]:
    name = str(getattr(camera_obj, "name", "Camera"))
    try:
        _, rig, sub = name.split("_")
        return int(rig), int(sub)
    except Exception:
        return (0, 0)


def _save_camera_parameters(camera_obj, output_folder: Path, frame: int):
    scene = bpy.context.scene
    if frame is not None:
        scene.frame_set(frame)
    # Ensure sensor aspect matches render aspect to avoid ValueError in K computation
    try:
        K = util_cam.get_calibration_matrix_K_from_blender(camera_obj.data)
    except Exception:
        try:
            W = scene.render.resolution_x
            H = scene.render.resolution_y
            if H > 0:
                camd = camera_obj.data
                # Adjust sensor_width to match render aspect while keeping height
                camd.sensor_width = float(camd.sensor_height) * (float(W) / float(H))
            K = util_cam.get_calibration_matrix_K_from_blender(camera_obj.data)
        except Exception:
            # Fallback to identity intrinsics if still failing
            K = np.eye(3, dtype=np.float64)
    HW = np.array((scene.render.resolution_y, scene.render.resolution_x))
    T = np.asarray(camera_obj.matrix_world, dtype=np.float64) @ np.diag(
        (1.0, -1.0, -1.0, 1.0)
    )
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    rig_id, subcam_id = _get_cam_ids(camera_obj)
    suffix = f"_{rig_id}_0_{frame}_{subcam_id}"
    np.savez(
        output_folder / f"camview{suffix}.npz",
        K=np.asarray(K, dtype=np.float64),
        T=T,
        HW=HW,
    )


@dataclass
class FrameOutputs:
    ply_path: Path
    scale_used: float
    points: int


def process_frame(
    scene, camera_obj, cfg: LidarConfig, out_dir: Path, frame: int
) -> FrameOutputs:
    """Emit LiDAR for a single frame and save PLY + metadata JSON lines."""
    assert bpy is not None, "Must run inside Blender"

    # Set frame and update depsgraph
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()

    # Rays in sensor frame
    rays = generate_sensor_rays(cfg)
    dirs_sensor = rays["directions"]  # (R,A,3), +X forward sensor frame
    rings = np.repeat(
        np.arange(dirs_sensor.shape[0], dtype=np.int16), dirs_sensor.shape[1]
    )
    az = np.tile(
        np.linspace(-math.pi, math.pi, dirs_sensor.shape[1], endpoint=False).astype(
            np.float32
        ),
        dirs_sensor.shape[0],
    )

    # World transform
    dirs_world = (
        _world_from_sensor(camera_obj, dirs_sensor).reshape(-1, 3).astype(np.float64)
    )
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
    meta = {
        "frame": frame,
        "points": int(pts_out.shape[0]),
        "scale_used": float(res.get("scale_used", 0.0)),
    }
    (out_dir / "frame_metadata.jsonl").write_text(
        (
            (out_dir / "frame_metadata.jsonl").read_text()
            if (out_dir / "frame_metadata.jsonl").exists()
            else ""
        )
        + json.dumps(meta)
        + "\n"
    )

    # Package camera intrinsics/extrinsics in a camview folder consistent with Infinigen
    rig_id, subcam_id = _get_cam_ids(camera_obj)
    camview_dir = out_dir / "camview" / f"camera_{rig_id}"
    camview_dir.mkdir(parents=True, exist_ok=True)
    try:
        _save_camera_parameters(camera_obj, camview_dir, frame=frame)
    except Exception:
        pass

    # Package IMU/TUM data: symlink/copy from sibling frames/imu_tum if present
    try:
        # seed root assumed to be input scene folder parent; infer from current .blend if available
        blend_path = (
            Path(bpy.data.filepath) if bpy and bpy.data and bpy.data.filepath else None
        )
        seed_root = blend_path.parent if blend_path is not None else out_dir.parent
        src_imu_dir = seed_root / "frames" / "imu_tum"
        dst_imu_dir = out_dir / "imu_tum"
        if src_imu_dir.exists():
            dst_imu_dir.mkdir(parents=True, exist_ok=True)
            # rig tum file is named like camrig_0_tum.txt
            rig_tum_name = f"camrig_{rig_id}_tum.txt"
            src_tum = src_imu_dir / rig_tum_name
            if src_tum.exists():
                dst_tum = dst_imu_dir / rig_tum_name
                if not dst_tum.exists():
                    try:
                        os.symlink(src_tum, dst_tum)
                    except Exception:
                        shutil.copy2(src_tum, dst_tum)
                # Convenience copy as poses_tum.txt at root of LiDAR folder
                poses_path = out_dir / "poses_tum.txt"
                if not poses_path.exists():
                    try:
                        os.symlink(dst_tum, poses_path)
                    except Exception:
                        shutil.copy2(dst_tum, poses_path)
    except Exception:
        pass

    # Write LiDAR calibration JSON (sensor<->camera mapping and key params)
    try:
        R_cs = sensor_to_camera_rotation().tolist()
        calib = {
            "frame_mode": getattr(cfg, "ply_frame", "sensor"),
            "sensor_to_camera_R_cs": R_cs,
            "min_range": float(getattr(cfg, "min_range", 0.05)),
            "max_range": float(getattr(cfg, "max_range", 100.0)),
            "azimuth_steps": int(
                getattr(cfg, "force_azimuth_steps", 0)
                or getattr(cfg, "azimuth_steps", 1800)
            ),
            "rings": int(getattr(cfg, "rings", 16)),
        }
        with (out_dir / "lidar_calib.json").open("w") as fh:
            json.dump(calib, fh, indent=2)
    except Exception:
        pass

    return FrameOutputs(
        out_ply, float(res.get("scale_used", 0.0)), int(pts_out.shape[0])
    )


# ------------------------- top-level entrypoints -------------------------


def _resolve_export_textures(scene_path: str | Path) -> str | None:
    try:
        s = Path(scene_path)
        scene_dir = s.parent
        scene_name = s.name
        candidates = [
            scene_dir / f"export_{scene_name}" / "textures",
            scene_dir / "export_scene.blend" / "textures",
            scene_dir.parent / "export" / f"export_{scene_name}" / "textures",
            scene_dir.parent / "export" / "textures",
        ]
        for c in candidates:
            if c.is_dir():
                return str(c)
        return None
    except Exception:
        return None


def run_on_current_scene(
    output_dir: str, frames: Sequence[int], camera_name: str, cfg_kwargs=None
):
    """Operate on the current open Blender scene."""
    assert bpy is not None, "Must run inside Blender"
    cfg_kwargs = dict(cfg_kwargs or {})
    cfg = LidarConfig(**cfg_kwargs)

    # Require Infinigen export-baked textures; try to auto-detect if not provided
    try:
        if not getattr(cfg, "export_bake_dir", None):
            blend_path = (
                Path(bpy.data.filepath)
                if bpy and bpy.data and bpy.data.filepath
                else None
            )
            if blend_path is not None:
                auto = _resolve_export_textures(blend_path)
                if auto:
                    cfg.export_bake_dir = auto
    except Exception:
        pass

    exp = getattr(cfg, "export_bake_dir", None)
    if not exp or not Path(exp).exists():
        raise FileNotFoundError(
            "LiDAR requires exporter-baked PBR textures. Provide --export-bake-dir or run scripts/bake_export_textures.sh"
        )

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
        random.seed(seed)
        np.random.seed(seed)

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


def generate_for_scene(
    scene_path: str,
    output_dir: str,
    frames: Sequence[int],
    camera_name="Camera",
    cfg_kwargs=None,
):
    """Open a .blend and run LiDAR generation."""
    assert bpy is not None, "Must run inside Blender"
    bpy.ops.wm.open_mainfile(filepath=str(scene_path))
    run_on_current_scene(
        output_dir=output_dir,
        frames=frames,
        camera_name=camera_name,
        cfg_kwargs=cfg_kwargs,
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser("Infinigen indoor LiDAR (baked-only)")
    p.add_argument("scene_path", type=str, help="Path to .blend")
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/infinigen_lidar",
        help="Output directory",
    )
    p.add_argument("--frames", type=str, default="1", help="e.g. '1-48' or '1,5,10'")
    p.add_argument("--camera", type=str, default="Camera", help="Camera object name")
    p.add_argument("--preset", type=str, default="VLP-16", help="Sensor preset")
    p.add_argument(
        "--force-azimuth-steps", type=int, default=None, help="Override azimuth columns"
    )
    p.add_argument(
        "--ply-frame",
        type=str,
        default="sensor",
        choices=["sensor", "camera", "world"],
        help="PLY output frame",
    )
    p.add_argument("--ply-binary", action="store_true", help="Write binary PLY")
    p.add_argument(
        "--auto-expose",
        action="store_true",
        help="Enable per-frame intensity auto-exposure (8-bit)",
    )
    p.add_argument(
        "--secondary",
        action="store_true",
        help="Enable single pass-through for transmissive surfaces",
    )
    p.add_argument("--seed", type=int, default=0)
    # Baked textures directory (required); auto-detected when present next to the scene
    p.add_argument(
        "--export-bake-dir",
        type=str,
        default=None,
        help="Folder containing exporter-baked PBR textures",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    args = parse_args(
        sys.argv[sys.argv.index("--") + 1 :]
        if argv is None and "--" in sys.argv
        else (argv or sys.argv[1:])
    )

    # Auto-detect baked textures dir if not provided. Check common exporter layouts:
    #  - <scene_dir>/export_<scene_name>/textures
    #  - <scene_dir>/export_scene.blend/textures (rare)
    #  - <scene_dir>/../export/export_<scene_name>/textures
    #  - <scene_dir>/../export/textures
    auto_bake = args.export_bake_dir or _resolve_export_textures(args.scene_path)

    cfg_kwargs = dict(
        preset=args.preset,
        force_azimuth_steps=args.force_azimuth_steps,
        ply_frame=args.ply_frame,
        enable_secondary=bool(args.secondary),
        auto_expose=bool(args.auto_expose),
        ply_binary=bool(args.ply_binary),
        prefer_ior=True,
        # Material sampling: baked-only (require exporter textures)
        export_bake_dir=auto_bake,
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
