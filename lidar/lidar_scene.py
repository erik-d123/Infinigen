#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# LiDAR scene utilities module
# Handles Blender scene setup and camera resolution

import sys
import bpy
from mathutils import Matrix

def setup_scene(scene_path: str):
    # Load Blender scene for ray casting
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.open_mainfile(filepath=scene_path)
    return bpy.context.scene

def resolve_camera(name: str | None):
    # Find specified camera or use first available
    if name:
        cam = bpy.data.objects.get(name)
        if cam:
            return cam
        print(f"Warning: camera '{name}' not found; using first camera.", file=sys.stderr)
    
    cams = [o for o in bpy.data.objects if o.type == 'CAMERA']
    if not cams:
        print("Error: no camera in scene.", file=sys.stderr)
        sys.exit(1)
    return cams[0]

def sensor_to_camera_rotation() -> Matrix:
    # Sensor frame (ROS): +X forward, +Y left, +Z up
    # Camera frame (Blender): +X right, +Y up, +Z backward (-Z forward)
    # Columns: sensor axes expressed in camera coordinates
    return Matrix(((0.0, -1.0,  0.0),
                   (0.0,  0.0,  1.0),
                   (-1.0, 0.0,  0.0)))
