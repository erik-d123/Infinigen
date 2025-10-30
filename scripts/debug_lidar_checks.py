#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ad-hoc diagnostics for LiDAR material handling.

Run with:
  Blender.app/Contents/MacOS/Blender --background --python scripts/debug_lidar_checks.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from lidar.intensity_model import extract_material_properties  # noqa: E402
from lidar.lidar_config import LidarConfig  # noqa: E402
from lidar.lidar_generator import process_frame  # noqa: E402
from lidar.lidar_raycast import generate_sensor_rays  # noqa: E402
from lidar.lidar_scene import sensor_to_camera_rotation  # noqa: E402


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 24
    return scene


def make_principled_material(
    name: str,
    *,
    base_color=(0.8, 0.8, 0.8, 1.0),
    metallic: float = 0.0,
    roughness: float = 0.2,
    specular: float = 0.5,
    transmission: float = 0.0,
    alpha: float | None = None,
):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = None
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            bsdf = node
            break
    if bsdf is None:
        raise RuntimeError("Failed to locate Principled BSDF")

    def _set_input(sock_name: str, value):
        sock = bsdf.inputs.get(sock_name)
        if sock is not None:
            sock.default_value = value

    _set_input("Base Color", base_color)
    _set_input("Metallic", metallic)
    _set_input("Roughness", roughness)

    if bsdf.inputs.get("Specular") is not None:
        _set_input("Specular", specular)
    else:
        _set_input("Specular IOR Level", specular)

    if bsdf.inputs.get("Transmission") is not None:
        _set_input("Transmission", transmission)
    else:
        _set_input("Transmission Weight", transmission)

    if alpha is not None:
        _set_input("Alpha", alpha)

    return mat


def make_plane(name: str, size: float = 5.0) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=size)
    plane = bpy.context.active_object
    plane.name = name
    return plane


def get_props_for_obj(obj: bpy.types.Object, poly_index: int = 0):
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return extract_material_properties(obj, poly_index, depsgraph)


def test_default_opacity():
    reset_scene()

    plane = make_plane("OpacityPlane")
    mat_opaque = make_principled_material("OpaqueMat")
    plane.data.materials.append(mat_opaque)
    props_default = get_props_for_obj(plane)

    mat_alpha = make_principled_material("AlphaMat", alpha=0.25)
    plane.data.materials[0] = mat_alpha
    props_alpha = get_props_for_obj(plane)

    return {
        "default_opacity": props_default["opacity"],
        "alpha_override": props_alpha["opacity"],
    }


def test_indoor_material(scene_path: Path):
    bpy.ops.wm.open_mainfile(filepath=str(scene_path))
    depsgraph = bpy.context.evaluated_depsgraph_get()

    for obj in bpy.data.objects:
        if obj.type != "MESH" or not obj.data.materials:
            continue
        poly_idx = 0
        props = extract_material_properties(obj, poly_idx, depsgraph)
        mat = obj.material_slots[0].material
        if mat is None:
            continue
        return {
            "object": obj.name,
            "material": mat.name,
            "opacity": props["opacity"],
            "metallic": props["metallic"],
            "diffuse_albedo": props["diffuse_albedo"],
        }
    raise RuntimeError("No mesh with material found in scene")


def setup_simple_lidar_scene():
    reset_scene()
    plane = make_plane("TestPlane", size=20.0)
    # Keep plane horizontal at Z=0.
    mat = make_principled_material(
        "TestMat",
        base_color=(0.6, 0.6, 0.6, 1.0),
        metallic=0.0,
        roughness=0.2,
        specular=0.5,
        transmission=0.0,
    )
    plane.data.materials.append(mat)

    cam = bpy.data.cameras.new("LiDARCAM")
    cam_obj = bpy.data.objects.new("LiDARCAM", cam)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = (0.0, 0.0, 3.0)
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.camera = cam_obj

    return plane, cam_obj


def lidar_config():
    cfg = LidarConfig(
        preset="VLP-16",
        force_azimuth_steps=720,
        save_ply=False,
        auto_expose=False,
    )
    cfg.save_ply = False
    cfg.auto_expose = False
    return cfg


def run_lidar_sample(cam_obj, cfg, frame: int, phase=0.0):
    scene = bpy.context.scene
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()

    sensor_R = sensor_to_camera_rotation()
    precomp = generate_sensor_rays(cfg)
    nhit, phase_out, res = process_frame(
        scene,
        cam_obj,
        frame,
        scene.render.fps,
        output_dir=os.path.join(REPO_ROOT, "tmp_lidar_debug"),
        cfg=cfg,
        sensor_R=sensor_R,
        precomp=precomp,
        phase_offset_rad=phase,
        write_ply=False,
    )
    return nhit, phase_out, res


def test_incidence_angle():
    _, cam = setup_simple_lidar_scene()
    cfg = lidar_config()

    cam.rotation_euler = (0.0, 0.0, 0.0)
    nhit_normal, phase, res_normal = run_lidar_sample(cam, cfg, frame=1)

    cam.rotation_euler = (math.radians(35.0), 0.0, 0.0)
    nhit_tilt, _, res_tilt = run_lidar_sample(cam, cfg, frame=1, phase=phase)

    refl_normal = res_normal.get("reflectivity") if res_normal else np.array([])
    refl_tilt = res_tilt.get("reflectivity") if res_tilt else np.array([])

    return {
        "hits_normal": nhit_normal,
        "hits_tilt": nhit_tilt,
        "mean_reflectance_normal": float(np.mean(refl_normal)) if refl_normal.size else 0.0,
        "mean_reflectance_tilt": float(np.mean(refl_tilt)) if refl_tilt.size else 0.0,
    }


def test_planar_animation():
    plane, cam = setup_simple_lidar_scene()
    cfg = lidar_config()

    plane.location = (0.0, 0.0, 0.0)
    plane.keyframe_insert(data_path="location", frame=1)
    plane.location = (2.0, 0.0, 0.0)
    plane.keyframe_insert(data_path="location", frame=2)

    hits = {}
    centroids = {}
    phase = 0.0
    for frame in (1, 2):
        nhit, phase, res = run_lidar_sample(cam, cfg, frame=frame, phase=phase)
        hits[frame] = nhit
        if res and res["points_world"].shape[0]:
            pts = np.array(res["points_world"])
            centroids[frame] = pts.mean(axis=0).tolist()
        else:
            centroids[frame] = None

    return {
        "hits": hits,
        "centroids": centroids,
    }


def test_transmissive_secondary():
    reset_scene()
    back_plane, cam = setup_simple_lidar_scene()
    cfg = lidar_config()
    cfg.enable_secondary = True

    glass_plane = make_plane("GlassPane", size=20.0)
    glass_plane.location = (0.0, 0.0, 0.05)
    glass_mat = make_principled_material(
        "GlassMaterial",
        base_color=(0.9, 0.9, 0.9, 1.0),
        metallic=0.0,
        roughness=0.05,
        specular=0.9,
        transmission=0.95,
    )
    glass_plane.data.materials.append(glass_mat)

    hits, phase, res = run_lidar_sample(cam, cfg, frame=1)
    max_returns = 0
    primary_mean = 0.0
    secondary_mean = 0.0
    secondary_count = 0
    ranges = []
    return_ids = []
    trans_list = []

    if res:
        num_returns = res.get("num_returns")
        if num_returns is not None and len(num_returns):
            max_returns = int(np.max(num_returns))
        ret_ids = res.get("return_id")
        refl = res.get("reflectivity")
        ranges_arr = res.get("range_m")
        trans_vals = res.get("transmittance")
        if ret_ids is not None and refl is not None and len(ret_ids) == len(refl):
            ret_ids = np.asarray(ret_ids)
            refl = np.asarray(refl)
            if np.any(ret_ids == 1):
                primary_mean = float(np.mean(refl[ret_ids == 1]))
            if np.any(ret_ids == 2):
                secondary_mean = float(np.mean(refl[ret_ids == 2]))
                secondary_count = int(np.sum(ret_ids == 2))
            return_ids = ret_ids.tolist()
            if ranges_arr is not None:
                ranges = np.asarray(ranges_arr).tolist()
            if trans_vals is not None:
                trans_list = np.asarray(trans_vals).tolist()

    return {
        "hits": hits,
        "max_returns": max_returns,
        "secondary_count": secondary_count,
        "primary_mean": primary_mean,
        "secondary_mean": secondary_mean,
        "ranges": ranges,
        "return_ids": return_ids,
        "transmittance": trans_list,
    }


def test_transmissive_material():
    """Compare opaque vs transmissive material reflectance."""
    reset_scene()
    plane, cam = setup_simple_lidar_scene()
    cfg = lidar_config()

    hits_opaque, phase, res_opaque = run_lidar_sample(cam, cfg, frame=1)

    glass_mat = make_principled_material(
        "GlassMat",
        base_color=(0.9, 0.9, 0.9, 1.0),
        metallic=0.0,
        roughness=0.05,
        specular=0.9,
        transmission=0.9,
    )
    plane.data.materials[0] = glass_mat
    bpy.context.view_layer.update()

    hits_glass, _, res_glass = run_lidar_sample(cam, cfg, frame=1, phase=phase)

    mean_opaque = float(np.mean(res_opaque.get("reflectivity"))) if res_opaque else 0.0
    mean_glass = float(np.mean(res_glass.get("reflectivity"))) if res_glass else 0.0

    return {
        "hits": (hits_opaque, hits_glass),
        "mean_reflectance": (mean_opaque, mean_glass),
    }


def main():
    results = []

    opacity_res = test_default_opacity()
    results.append(("Opacity defaults", opacity_res))

    sample_blend = next(REPO_ROOT.glob("outputs/**/*.blend"), None)
    if sample_blend:
        indoor_res = test_indoor_material(sample_blend)
        results.append(("Indoor material sample", {"blend": str(sample_blend), **indoor_res}))
    else:
        results.append(("Indoor material sample", {"error": "No .blend found under outputs/"}))

    angle_res = test_incidence_angle()
    results.append(("Incidence vs reflectance", angle_res))

    anim_res = test_planar_animation()
    results.append(("Planar animation", anim_res))

    print("\n==== LiDAR Debug Results ====")
    for title, data in results:
        print(f"\n[{title}]")
        for key, value in data.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
