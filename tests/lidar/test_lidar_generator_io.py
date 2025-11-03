#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json

import numpy as np
import pytest

try:
    import bpy  # noqa: F401
except Exception:  # pragma: no cover
    bpy = None
    pytest.skip("LiDAR Blender integration tests require Blender (bpy)", allow_module_level=True)

from lidar.lidar_config import LidarConfig
from lidar.lidar_generator import process_frame
from tests.lidar._bake_utils import bake_current_scene


def _reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x = 320
    scene.render.resolution_y = 240
    return scene


def _make_plane(name="Plane", z=0.0):
    bpy.ops.mesh.primitive_plane_add(size=5.0, location=(0.0, 0.0, z))
    plane = bpy.context.active_object
    plane.name = name
    mat = bpy.data.materials.new(name=f"{name}_Mat")
    mat.use_nodes = True
    plane.data.materials.append(mat)
    return plane


def _make_camera():
    camd = bpy.data.cameras.new("LiDARCAM")
    cam = bpy.data.objects.new("LiDARCAM", camd)
    bpy.context.collection.objects.link(cam)
    cam.location = (0.0, 0.0, 3.0)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.camera = cam
    return cam


def test_process_frame_writes_outputs(tmp_path: Path):
    scene = _reset_scene()
    _ = _make_plane()
    cam = _make_camera()
    out_dir = tmp_path / "lidar"
    cfg = LidarConfig(preset="VLP-16")
    cfg.auto_expose = False
    # Bake textures for current scene and point LiDAR to them
    tex_dir = bake_current_scene(tmp_path, res=64)
    cfg.export_bake_dir = str(tex_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fo = process_frame(scene, cam, cfg, out_dir, frame=1)
    # PLY exists
    ply_path = out_dir / "lidar_frame_0001.ply"
    assert ply_path.exists()
    # camview exists
    camview_dir = out_dir / "camview" / "camera_0"
    npzs = sorted(camview_dir.glob("camview_*.npz"))
    assert npzs, "camview npz not written"
    # calib JSON exists
    calib = json.loads((out_dir / "lidar_calib.json").read_text())
    assert "sensor_to_camera_R_cs" in calib
