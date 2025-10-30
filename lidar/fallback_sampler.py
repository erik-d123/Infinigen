from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import bpy  # noqa: F401
except Exception:  # pragma: no cover
    bpy = None

from lidar.mesh_uv import hit_uv
from lidar.tbn import world_shading_normal_from_tangent_map


IMAGE_CACHE: dict = {}


def _get_image_array(img):
    key = getattr(img, "name", None)
    w, h = getattr(img, "size", (0, 0))
    if key in IMAGE_CACHE:
        cached = IMAGE_CACHE[key]
        if cached["size"] == (w, h):
            return cached["arr"], w, h
    px = np.array(img.pixels[:], dtype=np.float32)
    arr = px.reshape((h, w, 4))
    IMAGE_CACHE[key] = {"arr": arr, "size": (w, h)}
    return arr, w, h


def _sample_image_rgba(img, uv):
    arr, w, h = _get_image_array(img)
    if w <= 0 or h <= 0:
        return (0.0, 0.0, 0.0, 1.0)
    u = float(uv[0]) % 1.0
    v = float(uv[1]) % 1.0
    ix = int(min(w - 1, max(0, round(u * (w - 1)))))
    iy = int(min(h - 1, max(0, round(v * (h - 1)))))
    r, g, b, a = arr[iy, ix]
    return (float(r), float(g), float(b), float(a))


def apply_image_overrides(
    obj,
    depsgraph,
    bsdf,
    poly_index: int,
    hit_world,
    props: Dict,
    use_normals: bool,
) -> None:
    if bpy is None:
        return
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    try:
        deps = bpy.context.evaluated_depsgraph_get()
    except Exception:
        deps = depsgraph
    eval_obj = obj.evaluated_get(deps)
    mesh = eval_obj.to_mesh()
    try:
        uv = hit_uv(eval_obj, mesh, poly_index, hit_world)
        if uv is None:
            uv = (0.5, 0.5)
        def from_img(sock_name: str):
            sock = bsdf.inputs.get(sock_name)
            if sock is None or not sock.is_linked:
                return None
            from_node = sock.links[0].from_node
            if getattr(from_node, "type", "") != "TEX_IMAGE" or from_node.image is None:
                return None
            return _sample_image_rgba(from_node.image, uv)

        rgba = from_img("Base Color")
        if rgba is not None:
            props["base_color"] = tuple(max(0.0, min(1.0, c)) for c in rgba[:3])

        for name, key in [("Roughness", "roughness"), ("Metallic", "metallic"), ("Transmission", "transmission")]:
            rgba = from_img(name)
            if rgba is not None:
                props[key] = max(0.0, min(1.0, float(rgba[0])))

        rgba = from_img("Alpha")
        if rgba is not None:
            a = rgba[3] if len(rgba) > 3 else rgba[0]
            props["opacity"] = max(0.0, min(1.0, float(a)))

        if use_normals:
            try:
                norm_in = bsdf.inputs.get("Normal")
                if norm_in is not None and norm_in.is_linked:
                    n_from = norm_in.links[0].from_node
                    img_node = None
                    strength = 1.0
                    if getattr(n_from, "type", "") == "NORMAL_MAP":
                        col = n_from.inputs.get("Color")
                        if col is not None and col.is_linked:
                            img_cand = col.links[0].from_node
                            if getattr(img_cand, "type", "") == "TEX_IMAGE" and img_cand.image is not None:
                                img_node = img_cand
                        try:
                            strength = float(n_from.inputs.get("Strength").default_value)
                        except Exception:
                            strength = 1.0
                    elif getattr(n_from, "type", "") == "TEX_IMAGE" and n_from.image is not None:
                        img_node = n_from
                    if img_node is not None:
                        r, g, b, _ = _sample_image_rgba(img_node.image, uv)
                        tn = (2.0 * float(r) - 1.0, 2.0 * float(g) - 1.0, 2.0 * float(b) - 1.0)
                        nx, ny = tn[0] * strength, tn[1] * strength
                        nz = max(0.0, min(1.0, (1.0 - strength) + strength * tn[2]))
                        nw = world_shading_normal_from_tangent_map(eval_obj, mesh, poly_index, hit_world, nx, ny, nz)
                        if nw is not None:
                            props["shading_normal_world"] = nw
            except Exception:
                pass
    finally:
        try:
            eval_obj.to_mesh_clear()
        except Exception:
            pass
