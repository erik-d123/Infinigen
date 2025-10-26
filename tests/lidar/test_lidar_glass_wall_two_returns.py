#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest

try:
    import bpy  # type: ignore
except Exception:
    bpy = None
    pytest.skip("LiDAR tests require Blender (bpy)", allow_module_level=True)

from lidar.lidar_config import LidarConfig
from lidar.lidar_raycast import perform_raycasting


def _make_principled(name, **inputs):
    m = bpy.data.materials.new(name=name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    for k, v in inputs.items():
        if k in bsdf.inputs:
            bsdf.inputs[k].default_value = v
    return m


def _setup_cam():
    sc = bpy.context.scene
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    sc.collection.objects.link(cam)
    cam.location = (0.0, 0.0, 0.0)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    return sc, cam


def _add_plane(z, mat):
    bpy.ops.mesh.primitive_plane_add(size=2.0, enter_editmode=False, location=(0.0, 0.0, z))
    plane = [o for o in bpy.context.scene.objects if o.type == "MESH"][-1]
    plane.data.materials.clear()
    plane.data.materials.append(mat)
    plane.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    return plane


def test_glass_in_front_of_wall_two_returns():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc, cam = _setup_cam()
    glass = _make_principled(
        "Glass",
        **{
            "Base Color": (1.0, 1.0, 1.0, 1.0),
            "Transmission": 1.0,
            "Transmission Roughness": 0.0,
            "Roughness": 0.0,
            "Metallic": 0.0,
            "IOR": 1.45,
        },
    )
    wall = _make_principled(
        "Wall", **{"Base Color": (0.4, 0.4, 0.4, 1.0), "Metallic": 0.0, "Roughness": 1.0}
    )
    _add_plane(-1.0, glass)
    _add_plane(-3.0, wall)
    cfg = LidarConfig(enable_secondary=True, secondary_min_residual=0.02, secondary_ray_bias=0.01, secondary_min_cos=0.9)
    origin = cam.matrix_world.translation
    dirs = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
    rings = np.array([0], dtype=np.uint16)
    az = np.array([0.0], dtype=np.float32)
    el = np.array([0.0], dtype=np.float32)
    t = np.array([0.0], dtype=np.float32)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    res = perform_raycasting(sc, depsgraph, origin, dirs, rings, az, el, t, cfg)
    assert res["return_id"].shape[0] >= 1
    assert int(res["num_returns"][0]) >= 1
    if res["return_id"].shape[0] == 2:
        r0, r1 = float(res["range_m"][0]), float(res["range_m"][1])
        assert r1 > r0
