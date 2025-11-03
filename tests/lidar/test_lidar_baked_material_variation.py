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
from ._bake_utils import bake_current_scene


def _reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 24
    return scene


def _make_plane_with_principled(name="Plane", size=5.0):
    bpy.ops.mesh.primitive_plane_add(size=size)
    plane = bpy.context.active_object
    plane.name = name
    mat = bpy.data.materials.new(name=f"{name}_Mat")
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    plane.data.materials.append(mat)
    return plane, mat, bsdf


def _make_camera():
    camd = bpy.data.cameras.new("LiDARCAM")
    cam = bpy.data.objects.new("LiDARCAM", camd)
    bpy.context.collection.objects.link(cam)
    cam.location = (0.0, 0.0, 3.0)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.camera = cam
    return cam


def _single_beam():
    origins = np.array([[0.0, 0.0, 3.0]], dtype=np.float64)
    directions = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
    rings = np.array([0], dtype=np.uint16)
    az = np.array([0.0], dtype=np.float32)
    return origins, directions, rings, az


def test_albedo_change_requires_rebake(tmp_path):
    scene = _reset_scene()
    plane, mat, bsdf = _make_plane_with_principled()
    _ = _make_camera()

    # Dark material
    bsdf.inputs["Base Color"].default_value = (0.2, 0.2, 0.2, 1.0)
    bpy.context.view_layer.update()
    tex_dir_1 = bake_current_scene(tmp_path / "bake1", res=64)

    cfg = LidarConfig()
    cfg.auto_expose = False
    cfg.export_bake_dir = str(tex_dir_1)

    o, d, r, a = _single_beam()
    deps = bpy.context.evaluated_depsgraph_get()
    res1 = perform_raycasting(scene, deps, o, d, r, a, cfg)
    assert res1["reflectivity"].size == 1
    refl1 = float(res1["reflectivity"][0])

    # Change material brighter -> must re-bake to see effect
    bsdf.inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1.0)
    bpy.context.view_layer.update()
    tex_dir_2 = bake_current_scene(tmp_path / "bake2", res=64)
    cfg2 = LidarConfig()
    cfg2.auto_expose = False
    cfg2.export_bake_dir = str(tex_dir_2)

    res2 = perform_raycasting(scene, deps, o, d, r, a, cfg2)
    assert res2["reflectivity"].size == 1
    refl2 = float(res2["reflectivity"][0])

    assert refl2 > refl1

