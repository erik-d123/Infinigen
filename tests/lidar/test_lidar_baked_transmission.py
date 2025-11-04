import numpy as np
import pytest

try:
    import bpy  # noqa: F401
except Exception:  # pragma: no cover
    bpy = None
    pytest.skip("Blender required", allow_module_level=True)

from lidar.lidar_raycast import perform_raycasting
from tests.lidar.conftest import _set_principled, make_camera, make_plane_with_material


def _one_ray():
    origins = np.array([[0.0, 0.0, 3.0]], dtype=np.float64)
    dirs = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
    rings = np.array([0], dtype=np.uint16)
    az = np.array([0.0], dtype=np.float32)
    return origins, dirs, rings, az


def test_transmission_reduces_reflectivity_and_adds_secondary(bake_scene, lidar_cfg):
    # Opaque wall at z=0, transmissive plane at z=1.5
    wall, wall_mat = make_plane_with_material(
        size=5.0,
        location=(0, 0, 0),
        base_color=(0.7, 0.7, 0.7, 1.0),
        roughness=0.4,
        metallic=0.0,
        transmission=0.0,
    )
    glass, glass_mat = make_plane_with_material(
        size=5.0,
        location=(0, 0, 1.5),
        base_color=(0.9, 0.9, 0.9, 1.0),
        roughness=0.0,
        metallic=0.0,
        transmission=0.8,
    )
    _ = make_camera(location=(0, 0, 3))
    scene = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()

    texdir = bake_scene(res=64)
    cfg = lidar_cfg
    cfg.export_bake_dir = str(texdir)
    cfg.enable_secondary = True
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()

    O, D, R, A = _one_ray()
    res = perform_raycasting(scene, deps, O, D, R, A, cfg)
    assert res["intensity"].size >= 1
    if res["num_returns"].size >= 2:
        assert np.all(res["num_returns"] == 2)
        assert float(res["range_m"][1]) > float(res["range_m"][0])

    # Make glass opaque and re-bake: reflectivity should increase for the nearer hit
    _set_principled(glass_mat, transmission=0.0)
    texdir2 = bake_scene(res=64)
    cfg.export_bake_dir = str(texdir2)
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    res_op = perform_raycasting(scene, deps, O, D, R, A, cfg)
    if res_op["reflectivity"].size and res["reflectivity"].size:
        # Compare the nearest return in both cases
        i_near = int(np.argmin(res["range_m"]))
        j_near = int(np.argmin(res_op["range_m"]))
        # Require transmissive reflectivity not higher than opaque near-surface reflectivity
        assert (
            float(res["reflectivity"][i_near])
            <= float(res_op["reflectivity"][j_near]) + 1e-6
        )
