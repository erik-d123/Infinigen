# SPDX-License-Identifier: MIT
# Scene helpers: camera resolution and sensor<->camera frame mapping

from __future__ import annotations
from typing import Tuple, Optional

import numpy as np

try:
    import bpy
except Exception:
    bpy = None


def sensor_to_camera_rotation() -> np.ndarray:
    """
    Return R_cs (camera <- sensor).
    Sensor frame: +X forward, +Y left, +Z up.
    Blender camera: +X right, +Y up, -Z forward.

    Mapping:
      sensor +X (forward) -> camera -Z
      sensor +Y (left)    -> camera -X
      sensor +Z (up)      -> camera +Y
    """
    R = np.array(
        [
            [0.0, -1.0,  0.0],  # camera X  <- { -Y_sensor }
            [0.0,  0.0,  1.0],  # camera Y  <- { +Z_sensor }
            [-1.0, 0.0,  0.0],  # camera Z  <- { -X_sensor }
        ],
        dtype=float,
    )
    # v_cam = R_cs @ v_sensor
    return R


def resolve_camera(scene, camera_name: Optional[str] = None):
    """
    Resolve a camera object:
      1) Named object if provided
      2) scene.camera if set
      3) First object of type 'CAMERA'
    """
    assert bpy is not None, "resolve_camera requires Blender"
    if camera_name:
        obj = scene.objects.get(camera_name)
        if obj and obj.type == "CAMERA":
            return obj
    if getattr(scene, "camera", None):
        return scene.camera
    for obj in scene.objects:
        if getattr(obj, "type", "") == "CAMERA":
            return obj
    raise RuntimeError("No camera found in scene")


def setup_scene(scene_path: str, camera_name: Optional[str] = None) -> Tuple[object, object]:
    """
    Open a .blend and return (scene, camera).
    """
    assert bpy is not None, "setup_scene requires Blender"
    bpy.ops.wm.open_mainfile(filepath=str(scene_path))
    sc = bpy.context.scene
    cam = resolve_camera(sc, camera_name)
    return sc, cam