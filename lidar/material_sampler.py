"""Sampling baked PBR maps (per object/material) at hit UVs.

Strict baked-only mode:
 - Per-hit signals come only from exporter-baked textures (no node evaluation).
 - Per-material metadata comes only from sidecar JSONs placed next to bakes.

Baked textures used:
 - DIFFUSE (RGB[A])  → base_color, coverage (A channel)
 - ROUGHNESS (R)
 - METAL (R)
 - TRANSMISSION (R)
 - Optional COVERAGE (R) overrides DIFFUSE alpha if present

Sidecar JSON per (object, material):
 - alpha_mode: "CLIP" | "BLEND" | "HASHED"
 - alpha_clip: float
 - ior (preferred) or specular (for F0)
 - transmission_roughness (optional)
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import bpy
import numpy as np

from lidar.mesh_uv import hit_uv as _hit_uv  # reuse


def _sample_px_bilinear(px: np.ndarray, uv: Tuple[float, float]) -> np.ndarray:
    """Bilinear sample RGBA at UV in [0, 1)."""
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
    # Optional coverage override; otherwise DIFFUSE alpha is used
    "Coverage": "COVERAGE",
}


class MaterialSampler:
    _inst: Optional["MaterialSampler"] = None

    def __init__(self):
        # Cache baked maps per (mesh_name, material_name, bake_dir)
        self.cache: Dict[Tuple[str, str, Optional[str]], Dict[str, np.ndarray]] = {}
        self.sidecars: Dict[Tuple[str, str, Optional[str]], Dict] = {}
        # per-frame mesh cache: {(obj_name): (eval_obj, mesh, frame_idx)}
        self.mesh_cache: Dict[str, Tuple[object, object, int]] = {}
        self.mesh_epoch: Optional[int] = None

    @classmethod
    def get(cls) -> "MaterialSampler":
        """Return the process‑global sampler instance (lazy‑initialized)."""
        if cls._inst is None:
            cls._inst = MaterialSampler()
        return cls._inst

    def _clean_name(self, name: str) -> str:
        return name.replace(" ", "_").replace(".", "_")

    def _load_png(self, path: str) -> Optional[np.ndarray]:
        """Load a PNG via Blender and return it as an HxWx4 float array.

        The Blender image is removed after reading to avoid accumulation.
        """
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

    def _load_sidecar(
        self, export_bake_dir: Optional[str], obj_name: str, mat_name: str
    ) -> Optional[Dict]:
        if export_bake_dir is None:
            return None
        base = self._clean_name(obj_name)
        mat = self._clean_name(mat_name)
        # Prefer object+material.json; fallback to object.json
        candidates = [
            os.path.join(export_bake_dir, f"{base}_{mat}.json"),
            os.path.join(export_bake_dir, f"{base}.json"),
        ]
        for c in candidates:
            try:
                if os.path.isfile(c):
                    with open(c, "r", encoding="utf-8") as fh:
                        return json.load(fh)
            except Exception:
                continue
        return None

    def _load_export_bakes(
        self, obj, export_bake_dir: Optional[str]
    ) -> Dict[str, np.ndarray]:
        """Load all known bake maps for an object from a textures directory."""
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

    def _ensure_mesh_epoch(self, frame_idx: int):
        if self.mesh_epoch is None:
            self.mesh_epoch = frame_idx
            return
        if self.mesh_epoch != frame_idx:
            # clear previous meshes
            try:
                for eval_obj, _, _ in self.mesh_cache.values():
                    try:
                        eval_obj.to_mesh_clear()  # type: ignore[attr-defined]
                    except Exception:
                        pass
            except Exception:
                pass
            self.mesh_cache.clear()
            self.mesh_epoch = frame_idx

    def sample_properties(
        self,
        obj,
        depsgraph,
        poly_index: int,
        hit_world,
        export_bake_dir: Optional[str] = None,
    ) -> Optional[Dict]:
        """Sample baked material properties at a world‑space hit location.

        Returns a dict subset of {base_color, roughness, metallic, transmission}
        when successful, otherwise None.
        """
        # Cache evaluated meshes per object per frame to avoid repeated to_mesh calls
        try:
            from bpy import context as _bpy_context  # type: ignore

            curr_frame = int(_bpy_context.scene.frame_current)  # type: ignore
        except Exception:
            curr_frame = -1
        self._ensure_mesh_epoch(curr_frame)

        obj_key = getattr(obj, "name", None) or str(id(obj))
        if obj_key in self.mesh_cache:
            eval_obj, mesh, _ = self.mesh_cache[obj_key]
        else:
            eval_obj = obj.evaluated_get(depsgraph)
            mesh = eval_obj.to_mesh()
            self.mesh_cache[obj_key] = (eval_obj, mesh, curr_frame)
        try:
            uv = _hit_uv(eval_obj, mesh, poly_index, hit_world)
            if uv is None:
                return None
            # find material of polygon
            poly = mesh.polygons[poly_index]
            mat_idx = int(getattr(poly, "material_index", 0))
            # Prefer evaluated object's material slots for consistency with evaluated mesh
            if hasattr(eval_obj, "material_slots") and len(eval_obj.material_slots) > 0:
                if mat_idx < len(eval_obj.material_slots):
                    mat = eval_obj.material_slots[mat_idx].material
                else:
                    mat = eval_obj.active_material
            else:
                mat = (
                    obj.material_slots[mat_idx].material
                    if (obj.material_slots and mat_idx < len(obj.material_slots))
                    else obj.active_material
                )
            if mat is None:
                return None
            mesh_name = getattr(obj.data, "name", "") or ""
            obj_name = getattr(obj, "name", "") or mesh_name
            mat_name = getattr(mat, "name", "") or ""
            key = (mesh_name, mat_name, export_bake_dir)
            if key not in self.cache:
                baked = self._load_export_bakes(obj, export_bake_dir)
                self.cache[key] = baked
            baked = self.cache[key]
            # Load sidecar once per (obj,mat,dir)
            if key not in self.sidecars:
                self.sidecars[key] = (
                    self._load_sidecar(export_bake_dir, obj_name, mat_name) or {}
                )
            sidecar = self.sidecars[key]
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

            def pick_coverage() -> float:
                # Prefer explicit COVERAGE bake
                px = baked.get("Coverage")
                if px is not None:
                    v = float(_sample_px_bilinear(px, uv)[0])
                    return max(0.0, min(1.0, v))
                # Fall back to DIFFUSE alpha channel
                px = baked.get("Base Color")
                if px is not None:
                    a = float(_sample_px_bilinear(px, uv)[3])
                    return max(0.0, min(1.0, a))
                # No coverage present in bakes → error (strict baked-only)
                raise ValueError(
                    f"Missing coverage (COVERAGE.png or DIFFUSE alpha) for {mesh_name}:{mat_name}"
                )

            out["base_color"] = pick_rgb("Base Color", (1.0, 1.0, 1.0))
            out["roughness"] = pick_scalar("Roughness", 0.5)
            out["metallic"] = pick_scalar("Metallic", 0.0)
            out["transmission"] = pick_scalar("Transmission", 0.0)
            out["coverage"] = pick_coverage()

            # Merge in sidecar metadata; require at least alpha semantics and a BRDF scalar
            if not sidecar:
                raise ValueError(
                    f"Missing sidecar JSON for {mesh_name}:{mat_name} in {export_bake_dir}"
                )
            # Normalize keys
            alpha_mode = str(sidecar.get("alpha_mode", "BLEND")).upper()
            alpha_clip = float(sidecar.get("alpha_clip", 0.5))
            out["alpha_mode"] = alpha_mode
            out["alpha_clip"] = alpha_clip
            # BRDF scalar: prefer ior, else specular
            if "ior" in sidecar:
                out["ior"] = float(sidecar["ior"])
            elif "specular" in sidecar:
                out["specular"] = float(sidecar["specular"])
            else:
                raise ValueError(
                    f"Sidecar missing 'ior' or 'specular' for {mesh_name}:{mat_name}"
                )
            if "transmission_roughness" in sidecar:
                out["transmission_roughness"] = float(sidecar["transmission_roughness"])

            return out
        except Exception:
            return None
