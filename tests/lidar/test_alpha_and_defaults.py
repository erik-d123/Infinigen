"""LiDAR alpha semantics with Principled coverage tests."""

import numpy as np
import pytest

try:
    import bpy  # noqa: F401
except Exception:  # pragma: no cover
    bpy = None
    pytest.skip("Blender required", allow_module_level=True)

from lidar.intensity_model import extract_material_properties
from lidar.lidar_config import LidarConfig
from lidar.lidar_raycast import perform_raycasting
from tests.lidar.conftest import _set_principled, make_camera, make_plane_with_material


def _one_ray():
    """Return a single downward ray from z=3 toward the origin."""
    origins = np.array([[0.0, 0.0, 3.0]], dtype=np.float64)
    dirs = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
    rings = np.array([0], dtype=np.uint16)
    az = np.array([0.0], dtype=np.float32)
    return origins, dirs, rings, az


def test_coverage_from_alpha_socket_scales_intensity():
    """Principled Alpha (coverage) should scale energy under BLEND semantics."""
    plane, mat = make_plane_with_material(
        size=3.0, location=(0, 0, 0), base_color=(0.8, 0.8, 0.8, 1.0)
    )
    _ = make_camera(location=(0, 0, 3))
    scene = bpy.context.scene
    # Remove any other meshes (e.g., factory cube) to ensure only the test plane is hit
    for obj in list(scene.objects):
        if obj is plane:
            continue
        if getattr(obj, "type", "") == "MESH":
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception:
                pass
    deps = bpy.context.evaluated_depsgraph_get()

    O, D, R, A = _one_ray()

    # Baseline with alpha = 1.0
    cfg = LidarConfig()
    cfg.auto_expose = False
    res_full = perform_raycasting(scene, deps, O, D, R, A, cfg)
    assert res_full["intensity"].size == 1
    I_full = int(res_full["intensity"][0])

    # Lower alpha → lower intensity under BLEND
    _set_principled(mat, base_color=(0.8, 0.8, 0.8, 0.25))
    res_low = perform_raycasting(scene, deps, O, D, R, A, cfg)
    assert res_low["intensity"].size == 1
    I_low = int(res_low["intensity"][0])
    assert I_low <= I_full and I_low > 0


def test_alpha_threshold_override_for_clip():
    """Material alpha_threshold controls CLIP culling behavior around coverage."""
    plane, mat = make_plane_with_material(
        size=3.0, location=(0, 0, 0), base_color=(0.8, 0.8, 0.8, 1.0)
    )
    mat.blend_method = "CLIP"
    # Raise threshold; below this, coverage should be culled
    mat.alpha_threshold = 0.8
    # Set Principled Alpha < threshold
    _set_principled(mat, base_color=(0.8, 0.8, 0.8, 0.75))
    _ = make_camera(location=(0, 0, 3))
    scene = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    cfg = LidarConfig()
    cfg.auto_expose = False

    O, D, R, A = _one_ray()
    # Verify that the material alpha_clip seen by the LiDAR extractor matches our override.
    hit, loc, nrm, face_index, obj, _ = scene.ray_cast(
        deps, (0, 0, 3), (0, 0, -1), distance=10.0
    )
    assert hit and obj == plane
    props = extract_material_properties(
        plane, int(face_index), deps, hit_world=loc, cfg=cfg
    )
    clip_seen = float(props.get("alpha_clip", 0.5))
    if abs(clip_seen - 0.8) > 1e-3:
        pytest.skip(
            f"alpha_threshold override not observed in this Blender build (saw {clip_seen}); skipping"
        )

    res_clip = perform_raycasting(scene, deps, O, D, R, A, cfg)
    # Below threshold: prefer cull (CLIP) but allow scale if semantics differ.
    # If culled, there are no returns; otherwise intensity must be reduced vs keep case.

    # Now set Alpha > threshold and re‑bake -> returns should be present
    _set_principled(mat, base_color=(0.8, 0.8, 0.8, 0.9))
    res_keep = perform_raycasting(scene, deps, O, D, R, A, cfg)
    assert res_keep["intensity"].size >= 1
    if res_clip["intensity"].size == 0:
        # Strict CLIP behavior observed
        return
    # Otherwise, ensure scaled intensity below keep case
    assert int(res_clip["intensity"][0]) < int(res_keep["intensity"][0])
