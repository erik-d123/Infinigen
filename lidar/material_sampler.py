"""
MaterialSampler: caches baked PBR maps per (object, material) and provides fast per-hit sampling.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import os
import numpy as np

try:
    import bpy
except Exception:
    bpy = None

from lidar.mesh_uv import compute_barycentric as _barycentric  # reuse
from lidar.mesh_uv import hit_uv as _hit_uv  # reuse


def _sample_px_bilinear(px: np.ndarray, uv: Tuple[float, float]) -> np.ndarray:
    """Bilinear sample RGBA at uv in [0,1)."""
    h, w, _ = px.shape
    if w <= 0 or h <= 0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    u = float(uv[0]) % 1.0
    v = float(uv[1]) % 1.0
    x = u * (w - 1)
    y = v * (h - 1)
    x0 = int(np.floor(x)); x1 = min(w - 1, x0 + 1)
    y0 = int(np.floor(y)); y1 = min(h - 1, y0 + 1)
    dx = x - x0; dy = y - y0
    c00 = px[y0, x0]
    c10 = px[y0, x1]
    c01 = px[y1, x0]
    c11 = px[y1, x1]
    c0 = c00 * (1.0 - dx) + c10 * dx
    c1 = c01 * (1.0 - dx) + c11 * dx
    c = c0 * (1.0 - dy) + c1 * dy
    return c


class MaterialSampler:
    _inst: Optional["MaterialSampler"] = None

    def __init__(self):
        # cache key includes optional export bake dir so we don't mix sources
        self.cache: Dict[Tuple[int, int, Optional[str]], Dict[str, np.ndarray]] = {}

    @classmethod
    def get(cls) -> "MaterialSampler":
        if cls._inst is None:
            cls._inst = MaterialSampler()
        return cls._inst

    def _clean_name(self, name: str) -> str:
        return name.replace(" ", "_").replace(".", "_")

    def _load_png(self, path: str) -> Optional[np.ndarray]:
        if bpy is None or not os.path.isfile(path):
            return None
        try:
            img = bpy.data.images.load(path, check_existing=True)
            w, h = img.size
            return np.array(img.pixels[:], dtype=np.float32).reshape((h, w, 4))
        except Exception:
            return None

    def _load_export_bakes(self, obj, export_bake_dir: Optional[str]) -> Dict[str, np.ndarray]:
        if export_bake_dir is None:
            return {}
        base = self._clean_name(obj.name)
        out: Dict[str, np.ndarray] = {}
        mapping = {
            'Base Color': 'DIFFUSE',
            'Roughness': 'ROUGHNESS',
            'Metallic': 'METAL',
            'Transmission': 'TRANSMISSION',
            'NormalTS': 'NORMAL',
        }
        for k, suf in mapping.items():
            p = os.path.join(export_bake_dir, f"{base}_{suf}.png")
            arr = self._load_png(p)
            if arr is not None:
                out[k] = arr
        return out

    def sample_properties(self, obj, depsgraph, poly_index: int, hit_world, res: int,
                          export_bake_dir: Optional[str] = None, use_export_bakes: bool = True) -> Optional[Dict]:
        if bpy is None:
            return None
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        try:
            uv = _hit_uv(eval_obj, mesh, poly_index, hit_world)
            if uv is None:
                return None
            # find material of polygon
            poly = mesh.polygons[poly_index]
            mat_idx = int(getattr(poly, 'material_index', 0))
            mat = obj.material_slots[mat_idx].material if (obj.material_slots and mat_idx < len(obj.material_slots)) else obj.active_material
            if mat is None:
                return None
            key = (id(obj.data), id(mat), export_bake_dir if use_export_bakes else None)
            if key not in self.cache:
                baked = self._load_export_bakes(obj, export_bake_dir) if use_export_bakes else {}
                self.cache[key] = baked
            baked = self.cache[key]
            out: Dict = {}
            def pick_scalar(name: str, default=0.0):
                px = baked.get(name)
                if px is None:
                    return default
                return float(_sample_px_bilinear(px, uv)[0])
            def pick_rgb(name: str, default=(1.0,1.0,1.0)):
                px = baked.get(name)
                if px is None:
                    return default
                r, g, b, _ = _sample_px_bilinear(px, uv)
                return (float(r), float(g), float(b))

            out['base_color'] = pick_rgb('Base Color', (1.0, 1.0, 1.0))
            out['roughness'] = pick_scalar('Roughness', 0.5)
            out['metallic'] = pick_scalar('Metallic', 0.0)
            out['transmission'] = pick_scalar('Transmission', 0.0)

            # Shading normal (tangent-space normal map) if present and enabled
            n_px = baked.get('NormalTS')
            if n_px is not None:
                r, g, b, _ = _sample_px_bilinear(n_px, uv)
                nx = 2.0 * float(r) - 1.0
                ny = 2.0 * float(g) - 1.0
                nz = 2.0 * float(b) - 1.0
                # Compute TBN
                mesh.calc_tangents()
                mesh.calc_loop_triangles()
                from mathutils import Vector
                M = np.array(eval_obj.matrix_world.to_3x3(), dtype=np.float64)
                # Find loop tri
                for tri in mesh.loop_triangles:
                    if tri.polygon_index != poly_index:
                        continue
                    l0, l1, l2 = tri.loops
                    vi0 = mesh.loops[l0].vertex_index
                    vi1 = mesh.loops[l1].vertex_index
                    vi2 = mesh.loops[l2].vertex_index
                    # bary for hit
                    M_inv = eval_obj.matrix_world.inverted()
                    hit_local = np.array(M_inv @ Vector(hit_world), dtype=np.float64)
                    a = np.array(mesh.vertices[vi0].co, dtype=np.float64)
                    b3 = np.array(mesh.vertices[vi1].co, dtype=np.float64)
                    c = np.array(mesh.vertices[vi2].co, dtype=np.float64)
                    u, v, w = _barycentric(hit_local, a, b3, c)
                    t0 = np.array(mesh.loops[l0].tangent, dtype=np.float64)
                    t1 = np.array(mesh.loops[l1].tangent, dtype=np.float64)
                    t2 = np.array(mesh.loops[l2].tangent, dtype=np.float64)
                    n0 = np.array(mesh.loops[l0].normal, dtype=np.float64)
                    n1 = np.array(mesh.loops[l1].normal, dtype=np.float64)
                    n2 = np.array(mesh.loops[l2].normal, dtype=np.float64)
                    sign0 = float(getattr(mesh.loops[l0], 'bitangent_sign', 1.0))
                    sign1 = float(getattr(mesh.loops[l1], 'bitangent_sign', 1.0))
                    sign2 = float(getattr(mesh.loops[l2], 'bitangent_sign', 1.0))
                    t = (u * t0 + v * t1 + w * t2)
                    n = (u * n0 + v * n1 + w * n2)
                    bvec = (u * (np.cross(n0, t0) * sign0) + v * (np.cross(n1, t1) * sign1) + w * (np.cross(n2, t2) * sign2))
                    # to world
                    T = M @ t; Nw = M @ n; B = M @ bvec
                    def _nz(x):
                        nrm = float(np.linalg.norm(x));
                        return x / nrm if nrm > 1e-12 else x
                    T = _nz(T); Nw = _nz(Nw); B = _nz(B)
                    nw = nx * T + ny * B + nz * Nw
                    nrm = float(np.linalg.norm(nw))
                    if nrm > 1e-12:
                        out['shading_normal_world'] = (nw / nrm).astype(np.float64)
                    break

            return out
        finally:
            try:
                eval_obj.to_mesh_clear()
            except Exception:
                pass
