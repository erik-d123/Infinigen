#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import pytest

try:
    import bpy  # type: ignore
except Exception:
    bpy = None
    pytest.skip("LiDAR tests require Blender (bpy)", allow_module_level=True)

from lidar.intensity_model import extract_material_properties, compute_intensity
from lidar.lidar_config import LidarConfig


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


def _setup_plane_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    sc.collection.objects.link(cam)
    cam.location = (0.0, 0.0, 0.0)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    bpy.ops.mesh.primitive_plane_add(size=2.0, enter_editmode=False, location=(0.0, 0.0, -2.0))
    plane = [o for o in bpy.context.scene.objects if o.type == "MESH"][-1]
    plane.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    return sc, cam, plane


def test_mirror_vs_diffuse_and_angle():
    sc, cam, plane = _setup_plane_scene()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mirror = _make_principled(
        "Mirror",
        **{"Base Color": (1.0, 1.0, 1.0, 1.0), "Metallic": 1.0, "Roughness": 0.0, "Specular": 0.5, "Transmission": 0.0},
    )
    diffuse = _make_principled(
        "Diffuse",
        **{"Base Color": (0.5, 0.5, 0.5, 1.0), "Metallic": 0.0, "Roughness": 1.0, "Specular": 0.0, "Transmission": 0.0},
    )
    cfg = LidarConfig()
    plane.data.materials.clear()
    plane.data.materials.append(mirror)
    props_m = extract_material_properties(plane, 0, depsgraph)
    I_m, _, _, _, _ = compute_intensity(props_m, cos_i=1.0, R=2.0, cfg=cfg)
    plane.data.materials[0] = diffuse
    bpy.context.view_layer.update()
    props_d = extract_material_properties(plane, 0, depsgraph)
    I_d, _, _, _, _ = compute_intensity(props_d, cos_i=1.0, R=2.0, cfg=cfg)
    assert I_m > I_d
    I_d_shallow, _, _, _, _ = compute_intensity(props_d, cos_i=math.cos(math.radians(60.0)), R=2.0, cfg=cfg)
    I_d_normal, _, _, _, _ = compute_intensity(props_d, cos_i=1.0, R=2.0, cfg=cfg)
    assert I_d_shallow < I_d_normal
