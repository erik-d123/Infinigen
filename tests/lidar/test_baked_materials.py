"""LiDAR material tests using Principled sampling."""

import numpy as np
import pytest

try:
    import bpy  # noqa: F401
except Exception:  # pragma: no cover
    bpy = None
    pytest.skip("Blender required", allow_module_level=True)

from infinigen.lidar.lidar_engine import (
    LidarConfig,
    extract_material_properties,
    perform_raycasting,
)
from tests.lidar.conftest import _set_principled, make_camera, make_plane_with_material


def _one_ray():
    """Return a single downward ray from z=3 toward the origin."""
    origins = np.array([[0.0, 0.0, 3.0]], dtype=np.float64)
    dirs = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
    rings = np.array([0], dtype=np.uint16)
    az = np.array([0.0], dtype=np.float32)
    return origins, dirs, rings, az


def test_albedo_change_updates_reflectivity():
    """Changing Principled albedo affects reflectivity immediately."""
    plane, mat = make_plane_with_material(
        size=2.0, location=(0, 0, 0), base_color=(0.2, 0.2, 0.2, 1.0)
    )
    _ = make_camera(location=(0, 0, 3))
    scene = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()

    cfg = LidarConfig()
    cfg.auto_expose = False
    O, D, R, A = _one_ray()
    res1 = perform_raycasting(scene, deps, O, D, R, A, cfg)
    assert res1["reflectivity"].size == 1
    refl1 = float(res1["reflectivity"][0])

    # Brighten
    _set_principled(mat, base_color=(0.9, 0.9, 0.9, 1.0))
    res2 = perform_raycasting(scene, deps, O, D, R, A, cfg)
    assert float(res2["reflectivity"][0]) >= refl1


def test_principled_property_extraction():
    """Sampler returns Principled BaseColor/Roughness/Metallic/Transmission per hit."""
    plane, mat = make_plane_with_material(
        size=2.0,
        location=(0, 0, 0),
        base_color=(0.3, 0.5, 0.7, 1.0),
        roughness=0.15,
        metallic=0.75,
    )
    _ = make_camera(location=(0, 0, 3))
    scene = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    cfg = LidarConfig()
    # Hit to get polygon index
    hit, loc, nrm, face_index, obj, _ = scene.ray_cast(
        deps, (0, 0, 3), (0, 0, -1), distance=10.0
    )
    assert hit and obj == plane
    props = extract_material_properties(
        plane, int(face_index), deps, hit_world=loc, cfg=cfg
    )
    assert (
        "base_color" in props
        and "roughness" in props
        and "metallic" in props
        and "transmission" in props
    )


def test_alpha_blend_and_clip():
    """BLEND scales energy by coverage; CLIP below threshold removes returns."""
    plane, mat = make_plane_with_material(
        size=2.0, location=(0, 0, 0), base_color=(0.8, 0.8, 0.8, 1.0)
    )
    _ = make_camera(location=(0, 0, 3))
    scene = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    O, D, R, A = _one_ray()

    # BLEND 0.5
    mat.blend_method = "BLEND"
    _set_principled(mat, base_color=(0.8, 0.8, 0.8, 0.5))
    cfg = LidarConfig()
    cfg.auto_expose = False
    res_blend = perform_raycasting(scene, deps, O, D, R, A, cfg)
    assert res_blend["intensity"].size >= 1

    # CLIP alpha below threshold → cull
    mat.blend_method = "CLIP"
    _set_principled(mat, base_color=(0.8, 0.8, 0.8, 0.25))
    res_clip = perform_raycasting(scene, deps, O, D, R, A, cfg)
    assert res_clip["intensity"].size == 0
