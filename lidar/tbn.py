"""
TBN utilities for converting tangent-space normal maps to world-space shading normals.
"""

from __future__ import annotations
from typing import Optional

import numpy as np

try:
    import bpy  # noqa: F401
    from mathutils import Vector  # type: ignore
except Exception:  # pragma: no cover
    bpy = None
    Vector = None  # type: ignore

from lidar.mesh_uv import compute_barycentric


def world_shading_normal_from_tangent_map(
    eval_obj,
    mesh,
    poly_index: int,
    hit_world,
    nx: float,
    ny: float,
    nz: float,
) -> Optional[np.ndarray]:
    """
    Convert a sampled tangent-space normal (nx,ny,nz in [-1,1]) at a hit point on a polygon
    into a world-space shading normal using per-loop TBN and barycentric interpolation.
    """
    if bpy is None or Vector is None:
        return None
    try:
        M_inv = eval_obj.matrix_world.inverted()
        hit_local = np.array(M_inv @ Vector(hit_world), dtype=np.float64)
    except Exception:
        return None

    mesh.calc_tangents()
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
        t0 = np.array(mesh.loops[l0].tangent, dtype=np.float64)
        t1 = np.array(mesh.loops[l1].tangent, dtype=np.float64)
        t2 = np.array(mesh.loops[l2].tangent, dtype=np.float64)
        n0 = np.array(mesh.loops[l0].normal, dtype=np.float64)
        n1 = np.array(mesh.loops[l1].normal, dtype=np.float64)
        n2 = np.array(mesh.loops[l2].normal, dtype=np.float64)
        s0 = float(getattr(mesh.loops[l0], "bitangent_sign", 1.0))
        s1 = float(getattr(mesh.loops[l1], "bitangent_sign", 1.0))
        s2 = float(getattr(mesh.loops[l2], "bitangent_sign", 1.0))
        t = (u * t0 + v * t1 + w * t2)
        n = (u * n0 + v * n1 + w * n2)
        bvec = (u * (np.cross(n0, t0) * s0) + v * (np.cross(n1, t1) * s1) + w * (np.cross(n2, t2) * s2))
        M = np.array(eval_obj.matrix_world.to_3x3(), dtype=np.float64)
        T = M @ t
        Nw = M @ n
        B = M @ bvec

        def _nz(x):
            nrm = float(np.linalg.norm(x))
            return x / nrm if nrm > 1e-12 else x

        T = _nz(T); Nw = _nz(Nw); B = _nz(B)
        nw = nx * T + ny * B + nz * Nw
        nrm = float(np.linalg.norm(nw))
        if nrm > 1e-12:
            return (nw / nrm).astype(np.float64)
        break
    return None

