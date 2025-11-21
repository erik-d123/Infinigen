"""LiDAR I/O and packaging tests.

Verifies that `process_frame` writes a PLY, camview package, and calibration
JSON when bakes are provided, using a small synthetic scene.
"""

import json

import pytest

try:
    import bpy  # noqa: F401
except Exception:  # pragma: no cover
    bpy = None
    pytest.skip("Blender required", allow_module_level=True)

from infinigen.lidar.lidar_engine import LidarConfig
from infinigen.lidar.lidar_generator import process_frame
from tests.lidar.conftest import make_camera, make_plane_with_material


def test_process_frame_writes_outputs(tmp_path):
    """End‑to‑end per‑frame run writes PLY, camview, and calibration JSON."""
    _ = make_plane_with_material(size=5.0, location=(0, 0, 0))
    cam = make_camera(location=(0, 0, 3))
    scene = bpy.context.scene

    out_dir = tmp_path / "lidar"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = LidarConfig(preset="VLP-16")
    cfg.auto_expose = False
    process_frame(scene, cam, cfg, out_dir, frame=1)

    # PLY exists
    assert (out_dir / "lidar_frame_0001.ply").exists()
    # camview exists
    camview_dir = out_dir / "camview" / "camera_0"
    npzs = sorted(camview_dir.glob("camview_*.npz"))
    assert npzs
    # calib JSON exists and has expected fields
    calib = json.loads((out_dir / "lidar_calib.json").read_text())
    assert "sensor_to_camera_R_cs" in calib and "rings" in calib
