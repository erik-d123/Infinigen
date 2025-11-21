import bpy
import numpy as np
import pytest

from infinigen.lidar import lidar_engine
from infinigen.lidar.lidar_engine import perform_raycasting


def _make_plane_with_bsdf(bsdf_type, location=(0, 0, 0), rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_plane_add(size=2, location=location, rotation=rotation)
    plane = bpy.context.active_object
    mat = bpy.data.materials.new(name=f"TestMat_{bsdf_type}")
    mat.use_nodes = True
    plane.data.materials.append(mat)

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new(bsdf_type)
    links.new(bsdf.outputs[0], output.inputs["Surface"])

    return plane, mat, bsdf


def _setup_camera(location=(0, 0, 3)):
    bpy.ops.object.camera_add(location=location, rotation=(0, 0, 0))
    cam = bpy.context.active_object
    bpy.context.scene.camera = cam
    return cam


@pytest.mark.parametrize(
    "bsdf_type, expected_behavior",
    [
        ("ShaderNodeBsdfGlass", "transmissive"),
        ("ShaderNodeBsdfDiffuse", "diffuse"),
        ("ShaderNodeBsdfGlossy", "specular"),
        ("ShaderNodeBsdfTransparent", "transparent"),
        ("ShaderNodeBsdfTranslucent", "diffuse_like"),
    ],
)
def test_bsdf_coverage(bsdf_type, expected_behavior, clear_sampler_cache):
    # Setup
    plane, mat, bsdf = _make_plane_with_bsdf(bsdf_type)
    cam = _setup_camera()
    assert cam is not None, "Camera not created"

    # Configure BSDF specific defaults to ensure signal
    if bsdf_type == "ShaderNodeBsdfGlass":
        bsdf.inputs["Roughness"].default_value = 0.0
        bsdf.inputs["IOR"].default_value = 1.45
    elif bsdf_type == "ShaderNodeBsdfDiffuse":
        bsdf.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.5
    elif bsdf_type == "ShaderNodeBsdfGlossy":
        bsdf.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.05
    elif bsdf_type == "ShaderNodeBsdfTransparent":
        bsdf.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)  # Fully transparent
    elif bsdf_type == "ShaderNodeBsdfTranslucent":
        bsdf.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)

    # Update
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()

    # Raycast
    cfg = lidar_engine.LidarConfig()
    O = np.array([[0, 0, 3]], dtype=np.float32)
    D = np.array([[0, 0, -1]], dtype=np.float32)
    R = np.array([3.0], dtype=np.float32)
    A = np.array([0], dtype=np.int64)

    res = perform_raycasting(bpy.context.scene, deps, O, D, R, A, cfg)

    # Assertions
    if expected_behavior == "transmissive":
        # Glass should have high transmittance
        assert res["transmittance"][0] > 0.9, "Glass should be highly transmissive"
        assert (
            res["reflectivity"][0] < 0.1
        ), "Glass should have low reflectivity at normal incidence"
    elif expected_behavior == "diffuse":
        # Diffuse should have 0 transmission and moderate reflectivity
        assert res["transmittance"][0] < 0.01, "Diffuse should be opaque"
        assert res["reflectivity"][0] > 0.1, "Diffuse should reflect light"
    elif expected_behavior == "specular":
        # Glossy should be highly reflective (metal-like in our mapping)
        assert res["reflectivity"][0] > 0.5, "Glossy should be highly reflective"
    elif expected_behavior == "transparent":
        # Transparent BSDF should be treated as fully transparent (culled primary).
        assert res["intensity"].size == 0
    elif expected_behavior == "diffuse_like":
        assert (
            res["transmittance"][0] < 0.01
        ), "Translucent treated as diffuse for reflection"
        assert res["reflectivity"][0] > 0.1
