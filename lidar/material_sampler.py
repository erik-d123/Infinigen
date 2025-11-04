"""
MaterialSampler: caches baked PBR maps per (object, material) and provides fast per-hit sampling.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import bpy
import numpy as np

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
    x0 = int(np.floor(x))
    x1 = min(w - 1, x0 + 1)
    y0 = int(np.floor(y))
    y1 = min(h - 1, y0 + 1)
    dx = x - x0
    dy = y - y0
    c00 = px[y0, x0]
    c10 = px[y0, x1]
    c01 = px[y1, x0]
    c11 = px[y1, x1]
    c0 = c00 * (1.0 - dx) + c10 * dx
    c1 = c01 * (1.0 - dx) + c11 * dx
    c = c0 * (1.0 - dy) + c1 * dy
    return c


BAKE_SUFFIX_MAP = {
    "Base Color": "DIFFUSE",
    "Roughness": "ROUGHNESS",
    "Metallic": "METAL",
    "Transmission": "TRANSMISSION",
}


class MaterialSampler:
    _inst: Optional["MaterialSampler"] = None

    def __init__(self):
        # Cache baked maps per (mesh_name, material_name, bake_dir)
        self.cache: Dict[Tuple[str, str, Optional[str]], Dict[str, np.ndarray]] = {}

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
            arr = np.array(img.pixels[:], dtype=np.float32).reshape((h, w, 4))
            # Free Blender image to avoid accumulating in bpy.data.images
            try:
                bpy.data.images.remove(img, do_unlink=True)
            except Exception:
                pass
            return arr
        except Exception:
            return None

    def _load_export_bakes(
        self, obj, export_bake_dir: Optional[str]
    ) -> Dict[str, np.ndarray]:
        if export_bake_dir is None:
            return {}
        base = self._clean_name(obj.name)
        out: Dict[str, np.ndarray] = {}
        for k, suf in BAKE_SUFFIX_MAP.items():
            p = os.path.join(export_bake_dir, f"{base}_{suf}.png")
            arr = self._load_png(p)
            if arr is not None:
                out[k] = arr
        return out

    def sample_properties(
        self,
        obj,
        depsgraph,
        poly_index: int,
        hit_world,
        export_bake_dir: Optional[str] = None,
    ) -> Optional[Dict]:
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        try:
            uv = _hit_uv(eval_obj, mesh, poly_index, hit_world)
            if uv is None:
                return None
            # find material of polygon
            poly = mesh.polygons[poly_index]
            mat_idx = int(getattr(poly, "material_index", 0))
            mat = (
                obj.material_slots[mat_idx].material
                if (obj.material_slots and mat_idx < len(obj.material_slots))
                else obj.active_material
            )
            if mat is None:
                return None
            mesh_name = getattr(obj.data, "name", "") or ""
            mat_name = getattr(mat, "name", "") or ""
            key = (mesh_name, mat_name, export_bake_dir)
            if key not in self.cache:
                baked = self._load_export_bakes(obj, export_bake_dir)
                self.cache[key] = baked
            baked = self.cache[key]
            out: Dict = {}

            def pick_scalar(name: str, default=0.0):
                px = baked.get(name)
                if px is None:
                    return default
                return float(_sample_px_bilinear(px, uv)[0])

            def pick_rgb(name: str, default=(1.0, 1.0, 1.0)):
                px = baked.get(name)
                if px is None:
                    return default
                r, g, b, _ = _sample_px_bilinear(px, uv)
                return (float(r), float(g), float(b))

            out["base_color"] = pick_rgb("Base Color", (1.0, 1.0, 1.0))
            out["roughness"] = pick_scalar("Roughness", 0.5)
            out["metallic"] = pick_scalar("Metallic", 0.0)
            out["transmission"] = pick_scalar("Transmission", 0.0)

            return out
        finally:
            try:
                eval_obj.to_mesh_clear()
            except Exception:
                pass
