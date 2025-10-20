#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import bpy

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
        frame = bpy.context.scene.frame_current
    except Exception:
        frame = None
    if frame != _CACHE_LAST_FRAME:
        _MATERIAL_CACHE.clear()
        _CACHE_LAST_FRAME = frame


def extract_material_properties(obj, poly_index, depsgraph):
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

    key = id(mat)
    cached = _MATERIAL_CACHE.get(key)
    if cached is not None:
        return cached

    params = dict(defaults)
    node = _find_principled_bsdf(mat)
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

    _MATERIAL_CACHE[key] = params
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
