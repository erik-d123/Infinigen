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


def test_cache_invalidation_across_frames():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    sc.collection.objects.link(cam)
    cam.location = (0.0, 0.0, 0.0)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    mat = _make_principled("Wall", **{"Base Color": (0.5, 0.5, 0.5, 1.0), "Roughness": 1.0, "Metallic": 0.0})
    bpy.ops.mesh.primitive_plane_add(size=2.0, enter_editmode=False, location=(0.0, 0.0, -2.0))
    plane = [o for o in bpy.context.scene.objects if o.type == "MESH"][-1]
    plane.data.materials.append(mat)
    bpy.context.view_layer.update()
    plane.keyframe_insert(data_path="location", frame=1)
    plane.location.z = -3.0
    plane.keyframe_insert(data_path="location", frame=2)
    bpy.context.view_layer.update()
    cfg = LidarConfig()
    origin = cam.matrix_world.translation
    dirs = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
    rings = np.array([0], dtype=np.uint16)
    az = np.array([0.0], dtype=np.float32)
    el = np.array([0.0], dtype=np.float32)
    t = np.array([0.0], dtype=np.float32)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    sc.frame_set(1)
    res1 = perform_raycasting(sc, depsgraph, origin, dirs, rings, az, el, t, cfg)
    sc.frame_set(2)
    res2 = perform_raycasting(sc, depsgraph, origin, dirs, rings, az, el, t, cfg)
    assert res1["range_m"][0] < res2["range_m"][0]
