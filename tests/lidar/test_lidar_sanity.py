#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pytest

try:
    import bpy  # noqa: F401
except ImportError:  # pragma: no cover - these tests require Blender's Python
    bpy = None
    pytest.skip("LiDAR sanity checks require Blender (bpy)", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.append(str(REPO_ROOT))

from scripts import debug_lidar_checks as lidar_debug


def assert_positive(value: float, name: str):
    assert value > 0.0, f"{name} should be positive, got {value}"


def test_default_opacity_values():
    """Ensure we get sensible defaults when Alpha socket is untouched."""
    res = lidar_debug.test_default_opacity()
    assert res["default_opacity"] == pytest.approx(1.0, abs=1e-6)
    assert res["alpha_override"] == pytest.approx(0.25, rel=1e-2)


def test_principled_material_property_extraction():
    """Procedurally create a principled material and ensure properties are parsed."""
    lidar_debug.reset_scene()
    plane = lidar_debug.make_plane("MatPlane")
    mat = lidar_debug.make_principled_material(
        "CustomMat",
        base_color=(0.3, 0.5, 0.7, 1.0),
        metallic=0.75,
        roughness=0.15,
        specular=0.65,
        transmission=0.1,
    )
    plane.data.materials.append(mat)
    props = lidar_debug.get_props_for_obj(plane)

    assert props["opacity"] == pytest.approx(1.0, abs=1e-6)
    assert props["metallic"] == pytest.approx(0.75, rel=1e-2)
    assert props["roughness"] == pytest.approx(0.15, rel=5e-2)
    assert 0.0 <= props["nir_reflectance"] <= 1.0
    assert props["F0_lum"] >= 0.02


def test_incidence_angle_reflectance_drop():
    """Tilting the LiDAR relative to a plane should reduce mean reflectance."""
    res = lidar_debug.test_incidence_angle()
    assert_positive(res["hits_normal"], "hits_normal")
    assert_positive(res["hits_tilt"], "hits_tilt")
    assert res["mean_reflectance_tilt"] < res["mean_reflectance_normal"]


def test_planar_animation_changes_point_centroid():
    """Moving a planar object should shift the LiDAR point cloud centroid."""
    res = lidar_debug.test_planar_animation()
    assert_positive(res["hits"][1], "frame_1_hits")
    assert_positive(res["hits"][2], "frame_2_hits")
    c1 = np.array(res["centroids"][1])
    c2 = np.array(res["centroids"][2])
    assert np.linalg.norm(c2 - c1) > 1e-3
    assert c2[0] > c1[0], "plane translation along +X should move centroid forward"


def test_material_change_affects_reflectance():
    """Swapping to a metallic material should increase backscatter reflectance."""
    lidar_debug.reset_scene()
    plane, cam = lidar_debug.setup_simple_lidar_scene()
    cfg = lidar_debug.lidar_config()

    hits_base, phase, res_base = lidar_debug.run_lidar_sample(cam, cfg, frame=1)
    assert_positive(hits_base, "baseline hits")
    assert np.all(np.asarray(res_base["range_m"]) > 0)
    refl_base = np.mean(res_base["return_power"])

    metallic_mat = lidar_debug.make_principled_material(
        "MetalMat",
        base_color=(0.8, 0.8, 0.8, 1.0),
        metallic=1.0,
        roughness=0.05,
        specular=0.9,
    )
    plane.data.materials[0] = metallic_mat
    bpy.context.view_layer.update()

    hits_metal, _, res_metal = lidar_debug.run_lidar_sample(cam, cfg, frame=1, phase=phase)
    assert_positive(hits_metal, "metal hits")
    refl_metal = np.mean(res_metal["return_power"])

    assert refl_metal > refl_base


def test_transmissive_material_reduces_reflectance():
    """Highly transmissive glass should reflect less than opaque baseline."""
    res = lidar_debug.test_transmissive_material()
    hits_opaque, hits_glass = res["hits"]
    assert_positive(hits_opaque, "opaque hits")
    assert_positive(hits_glass, "glass hits")
    refl_opaque, refl_glass = res["mean_reflectance"]
    assert refl_glass < refl_opaque


def test_transmissive_secondary_returns():
    """Glass pane in front of opaque surface should produce a second weaker return."""
    res = lidar_debug.test_transmissive_secondary()
    assert_positive(res["hits"], "glass scene hits")
    assert res["max_returns"] >= 2
    assert res["secondary_count"] > 0
    assert res["secondary_mean"] < res["primary_mean"]
    ranges = np.asarray(res.get("ranges", []), dtype=float)
    ids = np.asarray(res.get("return_ids", []), dtype=float)
    if ranges.size and ids.size:
        assert ranges.shape[0] == ids.shape[0]
    if res.get("transmittance") is not None:
        trans = np.asarray(res["transmittance"], dtype=float)
        assert np.all(trans >= 0.0)
