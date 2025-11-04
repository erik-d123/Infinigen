"""
Shared helpers to compute barycentric UVs for a hit on a mesh polygon.
"""

from __future__ import annotations

from typing import Optional, Tuple

import bpy  # noqa: F401
import numpy as np
from mathutils import Vector  # type: ignore


def compute_barycentric(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> Tuple[float, float, float]:
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return (1.0, 0.0, 0.0)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return (float(u), float(v), float(w))


def hit_uv(eval_obj, mesh, poly_index: int, hit_world) -> Optional[Tuple[float, float]]:
    uv_layer = getattr(mesh.uv_layers, "active", None)
    if uv_layer is None:
        return None
    uv_data = uv_layer.data
    try:
        M_inv = eval_obj.matrix_world.inverted()
        hit_local_v = M_inv @ Vector(hit_world)
        hit_local = np.array(hit_local_v, dtype=np.float64)
    except Exception:
        return None
    mesh.calc_loop_triangles()
    for tri in mesh.loop_triangles:
        if tri.polygon_index != poly_index:
            continue
        l0, l1, l2 = tri.loops
        vi0 = mesh.loops[l0].vertex_index
        vi1 = mesh.loops[l1].vertex_index
        vi2 = mesh.loops[l2].vertex_index
        a = np.array(mesh.vertices[vi0].co, dtype=np.float64)
        b = np.array(mesh.vertices[vi1].co, dtype=np.float64)
        c = np.array(mesh.vertices[vi2].co, dtype=np.float64)
        u, v, w = compute_barycentric(hit_local, a, b, c)
        uv0 = np.array(uv_data[l0].uv, dtype=np.float64)
        uv1 = np.array(uv_data[l1].uv, dtype=np.float64)
        uv2 = np.array(uv_data[l2].uv, dtype=np.float64)
        uv = u * uv0 + v * uv1 + w * uv2
        return (float(uv[0]), float(uv[1]))
    return None
