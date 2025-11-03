#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest

try:
    import bpy  # noqa: F401
except Exception:  # pragma: no cover
    bpy = None
    pytest.skip("LiDAR Blender integration tests require Blender (bpy)", allow_module_level=True)

from lidar.lidar_config import LidarConfig
from lidar.lidar_raycast import perform_raycasting
from tests.lidar._bake_utils import bake_current_scene


def _reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 24
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


def test_animation_variation_changes_range(tmp_path):
    scene = _reset_scene()
    plane = _make_plane(z=0.0)
    cam = _make_camera()
    deps = bpy.context.evaluated_depsgraph_get()

    # Fixed single beam straight down
    origins = np.array([[0.0, 0.0, 3.0]], dtype=np.float64)
    directions = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
    rings = np.array([0], dtype=np.uint16)
    az = np.array([0.0], dtype=np.float32)

    cfg = LidarConfig()
    cfg.auto_expose = False
    tex_dir = bake_current_scene(tmp_path, res=64)
    cfg.export_bake_dir = str(tex_dir)

    # Frame 1: plane at z=0
    scene.frame_set(1)
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    res1 = perform_raycasting(scene, deps, origins, directions, rings, az, cfg)

    # Frame 2: move plane further by 0.5 m towards -Z
    plane.location.z = -0.5
    scene.frame_set(2)
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    res2 = perform_raycasting(scene, deps, origins, directions, rings, az, cfg)

    assert res1["range_m"].size == 1 and res2["range_m"].size == 1
    assert res2["range_m"][0] > res1["range_m"][0]
    # With auto_expose off, farther should not be brighter
    if res1["intensity"].size and res2["intensity"].size:
        assert int(res2["intensity"][0]) <= int(res1["intensity"][0])
