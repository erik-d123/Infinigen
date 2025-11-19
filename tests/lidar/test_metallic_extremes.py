"""LiDAR metallic extremes tests.

Compare dielectric vs fully metallic surfaces at shallow roughness to ensure
the specular‑dominated metallic case yields higher reflectivity than the
dielectric with identical base color and pose.
"""

import numpy as np
import pytest

try:
    import bpy  # noqa: F401
except Exception:  # pragma: no cover
    bpy = None
    pytest.skip("Blender required", allow_module_level=True)

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


def test_metallic_vs_dielectric_reflectivity():
    """Metallic (m=1, low roughness) should be more reflective than dielectric."""
    plane, mat = make_plane_with_material(
        size=3.0,
        location=(0, 0, 0),
        base_color=(0.9, 0.9, 0.9, 1.0),
        roughness=0.05,
        metallic=0.0,
    )
    _ = make_camera(location=(0, 0, 3))
    scene = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    O, D, R, A = _one_ray()

    # Dielectric bake
    cfg = LidarConfig()
    cfg.auto_expose = False
    res_die = perform_raycasting(scene, deps, O, D, R, A, cfg)
    assert res_die["reflectivity"].size == 1
    R_die = float(res_die["reflectivity"][0])

    # Switch to metallic and re‑bake
    _set_principled(mat, metallic=1.0, roughness=0.05)
    res_met = perform_raycasting(scene, deps, O, D, R, A, cfg)
    assert res_met["reflectivity"].size == 1
    R_met = float(res_met["reflectivity"][0])

    assert R_met >= R_die - 1e-6
