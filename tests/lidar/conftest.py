"""Local helpers and fixtures for Principled-first LiDAR tests."""

from __future__ import annotations

import pytest

try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover
    bpy = None
    pytest.skip("LiDAR Blender tests require Blender (bpy)", allow_module_level=True)

from lidar.lidar_config import LidarConfig


def _set_principled(
    material,
    *,
    base_color=(0.8, 0.8, 0.8, 1.0),
    roughness=0.3,
    metallic=0.0,
    transmission=0.0,
    alpha=None,
    blend_method=None,
):
    material.use_nodes = True
    nt = material.node_tree
    bsdf = next(
        (n for n in nt.nodes if getattr(n, "type", "") == "BSDF_PRINCIPLED"), None
    )
    if bsdf is None:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    rgba = list(base_color)
    if alpha is not None and len(rgba) == 4:
        rgba[3] = float(alpha)
    bsdf.inputs["Base Color"].default_value = tuple(rgba)
    # Ensure Principled Alpha socket reflects requested alpha (CLIP semantics rely on this)
    alpha_val = (
        float(alpha)
        if alpha is not None
        else (float(rgba[3]) if len(rgba) == 4 else None)
    )
    if alpha_val is not None and "Alpha" in bsdf.inputs:
        try:
            bsdf.inputs["Alpha"].default_value = alpha_val
        except Exception:
            pass
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = float(roughness)
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = float(metallic)
    # Transmission name varies by Principled version
    if "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = float(transmission)
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = float(transmission)
    if blend_method is not None:
        material.blend_method = str(blend_method).upper()


def make_plane_with_material(
    *,
    size=2.0,
    location=(0, 0, 0),
    rotation=(0, 0, 0),
    base_color=(0.8, 0.8, 0.8, 1.0),
    roughness=0.3,
    metallic=0.0,
    transmission=0.0,
    alpha=None,
    blend_method=None,
):
    bpy.ops.mesh.primitive_plane_add(
        size=float(size), location=tuple(location), rotation=tuple(rotation)
    )
    plane = bpy.context.active_object
    mat = bpy.data.materials.new(name="Mat")
    plane.data.materials.clear()
    plane.data.materials.append(mat)
    _set_principled(
        mat,
        base_color=base_color,
        roughness=roughness,
        metallic=metallic,
        transmission=transmission,
        alpha=alpha,
        blend_method=blend_method,
    )
    return plane, mat


def make_camera(*, location=(0, 0, 3), rotation=(0, 0, 0)):
    camd = bpy.data.cameras.new("C")
    cam = bpy.data.objects.new("C", camd)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.location = tuple(location)
    cam.rotation_euler = tuple(rotation)
    return cam


@pytest.fixture
def lidar_cfg():
    cfg = LidarConfig()
    cfg.auto_expose = False
    # keep deterministic if these knobs exist
    if hasattr(cfg, "noise_sigma"):
        cfg.noise_sigma = 0.0
    if hasattr(cfg, "range_jitter"):
        cfg.range_jitter = 0.0
    cfg.enable_secondary = True
    return cfg
