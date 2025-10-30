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
from lidar.intensity_model import extract_material_properties


def _reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def _make_plane_with_principled(name="Plane"):
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
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


def test_enable_image_fallback_samples_basecolor(tmp_path):
    scene = _reset_scene()
    plane, mat, bsdf = _make_plane_with_principled()
    cam = _make_camera()
    deps = bpy.context.evaluated_depsgraph_get()

    # 1x1 image with known color
    img = bpy.data.images.new("UnitColor", width=1, height=1)
    img.pixels[:] = [0.1, 0.7, 0.2, 1.0]  # RGBA
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = img
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

    # Raycast one ray to get polygon index/hit location for extraction sampling
    ray_origin = (0.0, 0.0, 3.0)
    ray_dir = (0.0, 0.0, -1.0)
    hit, loc, nrm, face_index, obj, _ = scene.ray_cast(deps, ray_origin, ray_dir, distance=10.0)
    assert hit and obj == plane

    cfg = LidarConfig()
    cfg.enable_image_fallback = True
    props = extract_material_properties(plane, int(face_index), deps, hit_world=loc, cfg=cfg)
    bc = tuple(props.get("base_color", (0, 0, 0)))
    # Blender may quantize image pixels to 8-bit; accept nearest 1/255 precision
    exp = tuple(round(c * 255) / 255.0 for c in (0.1, 0.7, 0.2))
    import pytest as _pytest
    assert bc == _pytest.approx(exp, rel=0, abs=2.0/255.0)
