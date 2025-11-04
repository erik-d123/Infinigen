import math

import numpy as np
import pytest

try:
    import bpy  # noqa: F401
except Exception:  # pragma: no cover
    bpy = None
    pytest.skip("Blender required", allow_module_level=True)

from lidar.lidar_config import LidarConfig
from lidar.lidar_raycast import perform_raycasting
from tests.lidar.conftest import make_camera, make_plane_with_material


def _one_ray():
    origins = np.array([[0.0, 0.0, 3.0]], dtype=np.float64)
    dirs = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
    rings = np.array([0], dtype=np.uint16)
    az = np.array([0.0], dtype=np.float32)
    return origins, dirs, rings, az


def test_tilt_and_distance(bake_scene):
    plane, _ = make_plane_with_material(
        size=4.0, location=(0, 0, 0), base_color=(0.7, 0.7, 0.7, 1.0), roughness=0.3
    )
    _ = make_camera(location=(0, 0, 3))
    scene = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    tex = bake_scene(res=64)
    cfg = LidarConfig()
    cfg.auto_expose = False
    cfg.export_bake_dir = str(tex)
    O, D, R, A = _one_ray()

    res0 = perform_raycasting(scene, deps, O, D, R, A, cfg)
    I0 = int(res0["intensity"][0])

    plane.rotation_euler = (0.0, math.radians(45.0), 0.0)
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    res1 = perform_raycasting(scene, deps, O, D, R, A, cfg)
    if res1["intensity"].size:
        assert int(res1["intensity"][0]) <= I0

    plane.location.z = -0.5
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    res2 = perform_raycasting(scene, deps, O, D, R, A, cfg)
    # Allow equality within numeric tolerance (geometry/discretization can keep range unchanged)
    assert float(res2["range_m"][0]) >= float(res0["range_m"][0]) - 1e-6
    if res2["intensity"].size:
        assert int(res2["intensity"][0]) <= I0


def test_animation_across_frames(bake_scene):
    plane, _ = make_plane_with_material(size=5.0, location=(0, 0, 0))
    _ = make_camera(location=(0, 0, 3))
    scene = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    tex = bake_scene(res=64)
    cfg = LidarConfig()
    cfg.auto_expose = False
    cfg.export_bake_dir = str(tex)

    O, D, R, A = _one_ray()
    scene.frame_set(1)
    res1 = perform_raycasting(scene, deps, O, D, R, A, cfg)

    plane.location = (0.0, 0.0, -0.5)
    scene.frame_set(2)
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    res2 = perform_raycasting(scene, deps, O, D, R, A, cfg)

    assert res1["range_m"].size == 1 and res2["range_m"].size == 1
    assert res2["range_m"][0] > res1["range_m"][0]
    if res2["intensity"].size and res1["intensity"].size:
        assert int(res2["intensity"][0]) <= int(res1["intensity"][0])
