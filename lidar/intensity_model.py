#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import bpy
from mathutils import geometry, Vector

_MATERIAL_CACHE: dict[int, dict] = {}
_CACHE_LAST_FRAME = None


def _find_principled_bsdf(mat: bpy.types.Material):
    if not mat or not mat.use_nodes or not mat.node_tree:
        return None
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    return None


def _safe_input(node, name, default):
    try:
        sock = node.inputs.get(name)
        if sock is None or sock.is_linked:
            return default
        value = sock.default_value
        if hasattr(value, "__len__"):
            if len(value) >= 3:
                return tuple(float(x) for x in value[:3])
            return float(value[0])
        return float(value)
    except Exception:
        return default


def _srgb_to_linear(c: float) -> float:
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _sample_image_pixel(image, uv):
    if not image or image.size[0] == 0 or image.size[1] == 0:
        return None
    if not image.pixels:
        try:
            image.pixels[0]
        except Exception:
            return None
    width, height = image.size
    channels = image.channels
    u = (uv.x % 1.0) * (width - 1)
    v = (uv.y % 1.0) * (height - 1)
    x = int(max(0, min(width - 1, round(u))))
    y = int(max(0, min(height - 1, round(v))))
    index = (y * width + x) * channels
    try:
        data = image.pixels[index : index + channels]
    except Exception:
        data = None
    if not data:
        return None
    return tuple(data)


def _barycentric_coords(tri_verts, point):
    a, b, c = tri_verts
    v0 = b - a
    v1 = c - a
    v2 = point - a
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return (u, v, w)


def _compute_hit_uv(eval_obj, mesh, poly_index, hit_world):
    if not mesh.uv_layers or not mesh.uv_layers.active:
        return None
    uv_layer = mesh.uv_layers.active
    inv_world = eval_obj.matrix_world.inverted()
    hit_local = inv_world @ hit_world
    mesh.calc_loop_triangles()
    for tri in mesh.loop_triangles:
        if tri.polygon_index != poly_index:
            continue
        verts = [mesh.vertices[i].co for i in tri.vertices]
        bary = _barycentric_coords(verts, hit_local)
        if bary is None:
            continue
        loops = tri.loops
        uv_vals = [uv_layer.data[loop_idx].uv.copy() for loop_idx in loops]
        uv = uv_vals[0] * bary[0] + uv_vals[1] * bary[1] + uv_vals[2] * bary[2]
        return uv
    return None


def _sample_socket_texture(node, socket_name, uv):
    if uv is None or node is None:
        return None
    sock = node.inputs.get(socket_name)
    if sock is None or not sock.is_linked:
        return None
    link = sock.links[0]
    from_node = link.from_node
    while hasattr(from_node, "inputs") and isinstance(from_node, bpy.types.NodeReroute):
        if not from_node.inputs[0].links:
            return None
        from_node = from_node.inputs[0].links[0].from_node
    if isinstance(from_node, bpy.types.ShaderNodeTexImage) and from_node.image:
        pixel = _sample_image_pixel(from_node.image, uv)
        if pixel is None:
            return None
        if socket_name == "Base Color":
            if from_node.image.colorspace_settings.name.lower().startswith("srgb"):
                return tuple(_srgb_to_linear(float(c)) for c in pixel[:3])
            return tuple(float(c) for c in pixel[:3])
        return float(pixel[0])
    return None


def _material_has_textures(node):
    if node is None:
        return False
    for name in ("Base Color", "Roughness", "Metallic", "Transmission", "Alpha"):
        sock = node.inputs.get(name)
        if sock and sock.is_linked:
            return True
    return False


def get_material_from_hit(obj: bpy.types.Object, poly_index: int, depsgraph) -> bpy.types.Material | None:
    try:
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.data
        if not hasattr(mesh, "polygons") or poly_index < 0:
            return None
        poly = mesh.polygons[poly_index]
        mats = mesh.materials
        if mats and poly.material_index < len(mats):
            return mats[poly.material_index]
        if obj.material_slots:
            return obj.material_slots[poly.material_index].material
    except Exception:
        return None
    return None


def _luma(rgb):
    r, g, b = rgb
    return max(0.0, min(1.0, 0.2126 * r + 0.7152 * g + 0.0722 * b))


def _maybe_clear_cache():
    global _CACHE_LAST_FRAME
    try:
        scene = bpy.context.scene
        frame = scene.frame_current
        subframe = getattr(scene, "frame_subframe", 0.0)
        frame_key = (frame, round(subframe, 6))
    except Exception:
        frame_key = None
    if frame_key != _CACHE_LAST_FRAME:
        _MATERIAL_CACHE.clear()
        _CACHE_LAST_FRAME = frame_key


def extract_material_properties(obj, poly_index, depsgraph, hit_world=None):
    _maybe_clear_cache()
    mat = get_material_from_hit(obj, poly_index, depsgraph)

    defaults = {
        "base_color": (0.8, 0.8, 0.8),
        "metallic": 0.0,
        "specular": 0.5,
        "roughness": 0.5,
        "transmission": 0.0,
        "ior": 1.45,
        "opacity": 1.0,
        "F0_lum": 0.04,
        "nir_reflectance": 0.6,
        "is_glass_hint": False,
    }

    if mat is None:
        return defaults

    params = dict(defaults)
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.data
    uv = _compute_hit_uv(eval_obj, mesh, poly_index, hit_world) if hit_world is not None else None

    node = _find_principled_bsdf(mat)
    has_textures = _material_has_textures(node)
    key = id(mat)
    if not has_textures:
        cached = _MATERIAL_CACHE.get(key)
        if cached is not None:
            return dict(cached)
    if node:
        params["base_color"] = _safe_input(node, "Base Color", params["base_color"])
        params["metallic"] = float(_safe_input(node, "Metallic", params["metallic"]))
        params["specular"] = float(_safe_input(node, "Specular", params["specular"]))
        params["roughness"] = float(_safe_input(node, "Roughness", params["roughness"]))
        params["transmission"] = float(_safe_input(node, "Transmission", params["transmission"]))
        try:
            params["ior"] = float(_safe_input(node, "IOR", params["ior"]))
        except Exception:
            params["ior"] = 1.45
        try:
            params["opacity"] = float(_safe_input(node, "Alpha", params["opacity"]))
        except Exception:
            params["opacity"] = 1.0
    elif hasattr(mat, "diffuse_color"):
        c = mat.diffuse_color
        params["base_color"] = (float(c[0]), float(c[1]), float(c[2]))

    try:
        name_l = (mat.name or "").lower()
        if any(k in name_l for k in ("glass", "window", "pane")):
            params["is_glass_hint"] = True
    except Exception:
        pass

    if uv is not None and node is not None:
        color_sample = _sample_socket_texture(node, "Base Color", uv)
        if color_sample is not None:
            params["base_color"] = tuple(max(0.0, min(1.0, float(c))) for c in color_sample[:3])
        rough_sample = _sample_socket_texture(node, "Roughness", uv)
        if rough_sample is not None:
            params["roughness"] = float(max(0.0, min(1.0, rough_sample)))
        metal_sample = _sample_socket_texture(node, "Metallic", uv)
        if metal_sample is not None:
            params["metallic"] = float(max(0.0, min(1.0, metal_sample)))
        trans_sample = _sample_socket_texture(node, "Transmission", uv)
        if trans_sample is not None:
            params["transmission"] = float(max(0.0, min(1.0, trans_sample)))
        alpha_sample = _sample_socket_texture(node, "Alpha", uv)
        if alpha_sample is not None:
            params["opacity"] = float(max(0.0, min(1.0, alpha_sample)))

    rough = max(0.0, min(1.0, float(params["roughness"])))
    params["roughness"] = rough

    metal = float(params["metallic"])
    ior = float(params["ior"])
    spec_slider = float(params["specular"])

    base_rgb = params["base_color"]
    base_luma = _luma(base_rgb)

    f0_from_spec = max(0.0, min(0.08, 0.08 * spec_slider))
    f0_from_ior = ((ior - 1.0) / (ior + 1.0)) ** 2 if ior else 0.04
    dielectric_f0 = max(0.02, min(0.95, max(f0_from_spec, f0_from_ior)))

    params["F0_dielectric"] = dielectric_f0
    params["base_luma"] = base_luma
    params["F0_lum"] = (1.0 - metal) * dielectric_f0 + metal * max(0.02, min(0.95, base_luma))

    nir = base_luma
    try:
        if hasattr(mat, "get") and mat.get("nir_reflectance") is not None:
            nir = float(mat["nir_reflectance"])
    except Exception:
        pass
    nir = max(0.05, min(0.9, nir))
    params["nir_reflectance"] = (nir ** 0.8) * (1.0 - min(1.0, metal))

    if not has_textures:
        _MATERIAL_CACHE[key] = dict(params)
    return params


def transmissive_reflectance(cos_i: float, ior: float) -> float:
    ior = float(ior if ior else 1.45)
    f0 = ((ior - 1.0) / (ior + 1.0)) ** 2
    cos_i = max(0.0, float(cos_i))
    F = f0 + (1.0 - f0) * (1.0 - cos_i) ** 5
    return float(F)


def _derive_f0_ior(specular: float, ior: float | None):
    if ior and ior > 1.0:
        F0 = ((ior - 1.0) / (ior + 1.0)) ** 2
        return F0, ior
    specular = max(0.0, min(1.0, specular))
    F0 = 0.08 * specular
    sqrtF0 = math.sqrt(max(1e-8, F0))
    n = (1.0 + sqrtF0) / max(1e-8, (1.0 - sqrtF0))
    return F0, float(n)


def compute_intensity(props: dict, cos_i: float, R: float, cfg):
    def _cfg(name, default):
        return getattr(cfg, name, default)

    dist_p = float(_cfg("distance_power", 2.0))
    beta = float(_cfg("beta_atm", 0.0))

    metallic = float(props.get("metallic", 0.0))
    rough = max(0.0, min(1.0, float(props.get("roughness", 0.5))))
    transmission = float(props.get("transmission", 0.0) or 0.0)
    opacity = float(props.get("opacity", 1.0) or 1.0)
    specular = float(props.get("specular", 0.5))
    base_rgb = props.get("base_color", (0.8, 0.8, 0.8))
    nir_reflectance = float(props.get("nir_reflectance", _luma(base_rgb)))

    cos_i = max(1e-4, float(cos_i))
    R = max(1e-3, float(R))

    trans_raw = max(transmission, 1.0 - opacity)
    trans_frac = max(0.0, min(1.0, trans_raw))

    F0_lum, ior = _derive_f0_ior(specular, float(props.get("ior", None)))
    if metallic >= 1.0:
        F0_lum = max(0.02, min(0.98, _luma(base_rgb)))

    a = max(1e-4, rough * rough)
    denom = (cos_i * cos_i * (a * a - 1.0) + 1.0)
    D = (a * a) / (math.pi * denom * denom)
    k = (a + 1.0) * (a + 1.0) / 8.0
    G1 = cos_i / (cos_i * (1.0 - k) + k)
    G = G1 * G1
    F = F0_lum + (1.0 - F0_lum) * (1.0 - cos_i) ** 5
    spec_term = (F * D * G) / (4.0 * cos_i * cos_i)

    diffuse = (1.0 - metallic) * (1.0 - trans_frac) * (nir_reflectance / math.pi) * cos_i
    specular_term = (1.0 - trans_frac) * spec_term

    reflectivity = diffuse + specular_term
    I = reflectivity / (R ** dist_p)
    if beta > 0.0:
        I *= math.exp(-2.0 * beta * R)

    Fsurf = transmissive_reflectance(cos_i, ior if ior else 1.45)
    residual_T = max(0.0, trans_frac * (1.0 - Fsurf))

    return float(I), float(residual_T)


def classify_material(props: dict) -> int:
    metal = float(props.get("metallic", 0.0))
    transmission = float(props.get("transmission", 0.0))
    opacity = float(props.get("opacity", 1.0))
    if metal >= 0.5:
        return 2
    if props.get("is_glass_hint", False) or max(transmission, 1.0 - opacity) >= 0.3:
        return 1
    return 0
