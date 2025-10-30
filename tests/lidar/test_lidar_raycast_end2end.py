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


def _reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 24
    return scene


def _make_plane(name="Plane", z=0.0, rot=(0.0, 0.0, 0.0)):
    bpy.ops.mesh.primitive_plane_add(size=5.0, location=(0.0, 0.0, z), rotation=rot)
    plane = bpy.context.active_object
    plane.name = name
    mat = bpy.data.materials.new(name=f"{name}_Mat")
    mat.use_nodes = True
    # Make transmissive optional by user later; default opaque
    plane.data.materials.append(mat)
    return plane, mat


def _make_camera():
    camd = bpy.data.cameras.new("LiDARCAM")
    cam = bpy.data.objects.new("LiDARCAM", camd)
    bpy.context.collection.objects.link(cam)
    cam.location = (0.0, 0.0, 3.0)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.camera = cam
    return cam


def test_intensity_vs_tilt_and_shapes(tmp_path):
    scene = _reset_scene()
    plane, mat = _make_plane()
    cam = _make_camera()
    deps = bpy.context.evaluated_depsgraph_get()

    # Single ray straight down -Z
    origins = np.array([[0.0, 0.0, 3.0]], dtype=np.float64)
    directions = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
    rings = np.array([0], dtype=np.uint16)
    az = np.array([0.0], dtype=np.float32)

    cfg = LidarConfig()
    cfg.auto_expose = False

    res_flat = perform_raycasting(scene, deps, origins, directions, rings, az, cfg)
    assert res_flat["points"].shape[1] == 3
    assert res_flat["intensity"].dtype == np.uint8
    assert res_flat["reflectivity"].dtype == np.float32

    # Tilt plane by 45 degrees around Y; incidence should decrease and intensity drop (if hit remains)
    plane.rotation_euler = (0.0, np.deg2rad(45.0), 0.0)
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    res_tilt = perform_raycasting(scene, deps, origins, directions, rings, az, cfg)
    # If a hit remains, intensity should be lower
    if res_flat["intensity"].size and res_tilt["intensity"].size:
        assert int(res_tilt["intensity"][0]) <= int(res_flat["intensity"][0])


def test_secondary_return_with_transmission(tmp_path):
    scene = _reset_scene()
    # Front plane transmissive at z=0; back plane opaque at z=-0.5
    front, m_front = _make_plane(name="Front", z=0.0)
    back, m_back = _make_plane(name="Back", z=-0.5)
    bsdf_front = next(n for n in m_front.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    tr = bsdf_front.inputs.get("Transmission") or bsdf_front.inputs.get("Transmission Weight")
    if tr is not None:
        tr.default_value = 0.9
    else:
        pytest.skip("Principled Transmission socket not available in this Blender build")
    cam = _make_camera()
    deps = bpy.context.evaluated_depsgraph_get()

    origins = np.array([[0.0, 0.0, 3.0]], dtype=np.float64)
    directions = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
    rings = np.array([0], dtype=np.uint16)
    az = np.array([0.0], dtype=np.float32)

    cfg = LidarConfig()
    cfg.enable_secondary = True
    cfg.auto_expose = False

    res = perform_raycasting(scene, deps, origins, directions, rings, az, cfg)
    # Expect either a merged stronger hit or two returns; check num_returns contains 2 or reflectivity non-zero
    assert res["intensity"].size > 0
    # If two returns recorded, both entries have num_returns==2
    if res["num_returns"].size >= 2:
        assert np.all(res["num_returns"] == 2)
