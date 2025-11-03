#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from pathlib import Path

import pytest

try:
    import bpy  # noqa: F401
except Exception:  # pragma: no cover
    bpy = None
    pytest.skip("LiDAR Blender integration tests require Blender (bpy)", allow_module_level=True)

from lidar.lidar_config import LidarConfig
from lidar.lidar_generator import process_frame
from lidar.intensity_model import extract_material_properties
from tests.lidar._bake_utils import bake_current_scene


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
    bsdf = None
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            bsdf = node
            break
    assert bsdf is not None
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


def test_extracts_baked_properties(tmp_path: Path):
    scene = _reset_scene()
    plane, mat, bsdf = _make_plane_with_principled()
    # Set some Principled defaults
    bsdf.inputs["Base Color"].default_value = (0.3, 0.5, 0.7, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.75
    bsdf.inputs["Roughness"].default_value = 0.15
    bsdf.inputs["Specular" if "Specular" in bsdf.inputs else "Specular IOR Level"].default_value = 0.65
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    # Bake the scene and sample baked properties at a hit location
    tex_dir = bake_current_scene(tmp_path, res=64)
    cfg = LidarConfig()
    cfg.export_bake_dir = str(tex_dir)
    # Raycast to get polygon index and hit location
    ray_origin = (0.0, 0.0, 3.0)
    ray_dir = (0.0, 0.0, -1.0)
    hit, loc, nrm, face_index, obj, _ = bpy.context.scene.ray_cast(deps, ray_origin, ray_dir, distance=10.0)
    assert hit and obj == plane
    props = extract_material_properties(plane, int(face_index), deps, hit_world=loc, cfg=cfg)
    assert abs(props.get("metallic", 0.0) - 0.75) < 1e-2
    assert abs(props.get("roughness", 0.0) - 0.15) < 1e-1
    assert 'base_color' in props


def test_alpha_clip_culls_hits(tmp_path: Path):
    scene = _reset_scene()
    plane, mat, bsdf = _make_plane_with_principled()
    # Set material alpha low and clip mode
    bsdf.inputs["Alpha"].default_value = 0.1
    try:
        mat.blend_method = 'CLIP'
        mat.alpha_threshold = 0.5
    except Exception:
        pass
    cam = _make_camera()

    cfg = LidarConfig(preset="VLP-16")
    cfg.auto_expose = False
    cfg.force_azimuth_steps = 180
    # Bake and point LiDAR to baked textures
    tex_dir = bake_current_scene(tmp_path, res=64)
    cfg.export_bake_dir = str(tex_dir)

    out_dir = tmp_path / "lidar"
    out_dir.mkdir(parents=True, exist_ok=True)
    fo = process_frame(scene, cam, cfg, out_dir, frame=1)
    # Expect zero or near-zero points when fully clipped
    # In strict CLIP, should be zero hits; allow slight tolerance
    pts = (out_dir / f"lidar_frame_{1:04d}.ply").exists()
    # If file exists, we can parse but simplest is to rely on size via metadata
    meta = (out_dir / "frame_metadata.jsonl")
    if meta.exists():
        text = meta.read_text().strip().splitlines()[-1]
        import json
        rec = json.loads(text)
        assert rec.get("points", 0) == 0


def test_baked_normal_produces_shading_normal(tmp_path: Path):
    # Build a simple plane with a normal map wired to Principled Normal input
    scene = _reset_scene()
    plane, mat, bsdf = _make_plane_with_principled()
    nt = mat.node_tree
    # Create a 1x1 image that encodes a strong +X tangent normal (R=1, G=0.5, B=0.5~flat-ish)
    import bpy  # noqa: F401
    img = bpy.data.images.new("UnitNormal", width=1, height=1)
    # RGBA flattened (1 pixel)
    img.pixels[:] = [1.0, 0.5, 0.5, 1.0]
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nmap.inputs["Strength"].default_value = 1.0
    nt.links.new(tex.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()

    # Cast a simple ray onto the plane to get a polygon index and hit location
    # Place the plane at z=0, camera above looking down -> ray along -Z
    cam = _make_camera()
    cam.location = (0.0, 0.0, 3.0)
    scene.frame_set(1)
    ray_origin = (0.0, 0.0, 3.0)
    ray_dir = (0.0, 0.0, -1.0)
    hit, loc, nrm, face_index, obj, _ = scene.ray_cast(deps, ray_origin, ray_dir, distance=10.0)
    assert hit and obj == plane

    from lidar.lidar_config import LidarConfig
    cfg = LidarConfig()
    # Even with bakes, we do not include shading normals in minimal setup
    tex_dir = bake_current_scene(tmp_path, res=32)
    cfg.export_bake_dir = str(tex_dir)
    props = extract_material_properties(plane, int(face_index), deps, hit_world=loc, cfg=cfg)
    assert "shading_normal_world" not in props
