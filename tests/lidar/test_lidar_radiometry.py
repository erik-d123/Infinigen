"""LiDAR radiometry and scene interaction tests."""

import math

import numpy as np
import pytest

try:
    import bpy
except ImportError:
    bpy = None
    pytest.skip("Blender required", allow_module_level=True)

from infinigen.lidar.lidar_engine import LidarConfig, perform_raycasting
from tests.lidar.conftest import _set_principled, make_camera, make_plane_with_material


def _one_ray_at(x, y, z, dx, dy, dz):
    origins = np.array([[x, y, z]], dtype=np.float64)
    dirs = np.array([[dx, dy, dz]], dtype=np.float64)
    # Normalize dir
    norm = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / norm
    rings = np.array([0], dtype=np.uint16)
    az = np.array([0.0], dtype=np.float32)
    return origins, dirs, rings, az


def test_incidence_angle_attenuation():
    """Reflectivity should decrease as incidence angle increases (away from normal)."""
    # Plane at origin, facing +Z
    plane, mat = make_plane_with_material(
        size=2.0,
        location=(0, 0, 0),
        rotation=(0, 0, 0),
        base_color=(0.8, 0.8, 0.8, 1.0),
        roughness=0.5,
    )
    _ = make_camera(location=(0, 0, 5))
    scene = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    cfg = LidarConfig()
    cfg.auto_expose = False

    # 1. Normal incidence (straight down)
    O, D, R, A = _one_ray_at(0, 0, 5, 0, 0, -1)
    res_normal = perform_raycasting(scene, deps, O, D, R, A, cfg)
    assert res_normal["intensity"].size == 1
    int_normal = float(res_normal["intensity"][0])

    # 2. Angled incidence (45 degrees)
    # Rotate plane by 45 deg around Y axis
    plane.rotation_euler = (0, math.radians(45), 0)
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()  # Update depsgraph

    res_angled = perform_raycasting(scene, deps, O, D, R, A, cfg)
    assert res_angled["intensity"].size == 1
    int_angled = float(res_angled["intensity"][0])

    print(f"Normal: {int_normal}, Angled: {int_angled}")
    assert int_angled < int_normal, "Intensity should drop at oblique angles"


def test_transmission_reduces_reflection():
    """Higher transmission should result in lower return intensity (light passes through)."""
    plane, mat = make_plane_with_material(
        size=2.0, location=(0, 0, 0), base_color=(1.0, 1.0, 1.0, 1.0), roughness=0.0
    )
    _ = make_camera(location=(0, 0, 5))
    scene = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    cfg = LidarConfig()
    cfg.auto_expose = False
    O, D, R, A = _one_ray_at(0, 0, 5, 0, 0, -1)

    # Opaque
    _set_principled(mat, transmission=0.0)
    res_opaque = perform_raycasting(scene, deps, O, D, R, A, cfg)
    int_opaque = float(res_opaque["intensity"][0])

    # Transmissive
    _set_principled(mat, transmission=0.9)
    res_trans = perform_raycasting(scene, deps, O, D, R, A, cfg)
    int_trans = float(res_trans["intensity"][0])

    print(f"Opaque: {int_opaque}, Transmissive: {int_trans}")
    assert int_trans < int_opaque, "Transmissive object should return less light"


def _add_image_texture(mat):
    """Add an image texture to the material's Base Color."""
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")

    # Create a 2x2 image: Left=Black, Right=White
    img = bpy.data.images.new("TestImg", width=2, height=1)
    # Pixels: R,G,B,A
    # Pixel 0 (Left): 0,0,0,1
    # Pixel 1 (Right): 1,1,1,1
    pixels = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    img.pixels = pixels

    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img

    # Explicitly use UV coords
    coord = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(coord.outputs["UV"], tex.inputs["Vector"])

    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])


def test_texture_variation():
    """Intensity should vary across a textured surface (Image Texture)."""
    plane, mat = make_plane_with_material(size=2.0, location=(0, 0, 0))
    _add_image_texture(mat)

    _ = make_camera(location=(0, 0, 5))
    scene = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    cfg = LidarConfig()
    cfg.auto_expose = False

    # Plane is 2x2, centered at 0. Extents: [-1, 1]
    # UVs usually map [0,1] to the mesh.
    # Left side (x < 0) -> UV u < 0.5 -> Black pixel
    # Right side (x > 0) -> UV u > 0.5 -> White pixel

    # Point 1: Left side (-0.5, 0)
    O1, D1, R1, A1 = _one_ray_at(-0.5, 0, 5, 0, 0, -1)
    res1 = perform_raycasting(scene, deps, O1, D1, R1, A1, cfg)
    int1 = float(res1["intensity"][0])

    # Point 2: Right side (0.5, 0)
    O2, D2, R2, A2 = _one_ray_at(0.5, 0, 5, 0, 0, -1)
    res2 = perform_raycasting(scene, deps, O2, D2, R2, A2, cfg)
    int2 = float(res2["intensity"][0])

    print(f"Left (Black): {int1}, Right (White): {int2}")
    assert int1 < int2, "Darker part of texture should return less intensity"


def test_moving_object():
    """LiDAR should miss the object when it moves out of the beam."""
    plane, mat = make_plane_with_material(
        size=1.0,
        location=(0, 0, 0),  # Small plane at origin
    )
    _ = make_camera(location=(0, 0, 5))
    scene = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    cfg = LidarConfig()
    O, D, R, A = _one_ray_at(0, 0, 5, 0, 0, -1)  # Aim at origin

    # 1. Hit
    res_hit = perform_raycasting(scene, deps, O, D, R, A, cfg)
    assert res_hit["intensity"].size == 1

    # 2. Move plane
    plane.location = (5, 0, 0)
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()

    # 3. Miss
    res_miss = perform_raycasting(scene, deps, O, D, R, A, cfg)
    assert res_miss["intensity"].size == 0
