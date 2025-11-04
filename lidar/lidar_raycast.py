# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.
# Indoor LiDAR raycasting loop with alpha-at-output and optional secondary

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import bpy
import numpy as np

from lidar.intensity_model import (
    classify_material,
    compute_intensity,
    extract_material_properties,
)

# ------------------------- helpers -------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _percentile_scale(raw_pos: np.ndarray, pct: float, target_u8: float) -> float:
    # Map the pct-th percentile of positive raw to target_u8/255
    p = float(np.percentile(raw_pos, pct))
    if p <= 1e-12:
        return 0.0
    return (target_u8 / 255.0) / p


def _compute_cos_i(normal: np.ndarray, ray_dir: np.ndarray) -> float:
    # ray_dir points from origin into scene. Incidence uses negative ray_dir.
    return float(max(0.0, min(1.0, -np.dot(_unit(ray_dir), _unit(normal)))))


# --------------------- public API: rays ---------------------


def generate_sensor_rays(cfg) -> Dict[str, np.ndarray]:
    """
    Build per-ring directions (sensor frame). Minimal, indoor defaults.
    Returns dict with:
      - directions: (R, A, 3) unit vectors in +X forward sensor frame
      - ring: (R,) ring indices
      - azimuth: (A,) azimuth samples in radians
    """
    rings = getattr(cfg, "rings", 16)
    az_steps = int(
        getattr(cfg, "force_azimuth_steps", 0) or getattr(cfg, "azimuth_steps", 1800)
    )
    # Elevation fan: linear indoor default if preset not provided
    elev = np.linspace(-15.0, 15.0, rings) * (math.pi / 180.0)
    az = np.linspace(-math.pi, math.pi, az_steps, endpoint=False)

    # Sensor frame: +X forward, +Y left, +Z up
    dirs = np.zeros((rings, az_steps, 3), dtype=np.float32)
    for r, el in enumerate(elev):
        ce, se = math.cos(el), math.sin(el)
        # Base forward points +X at az=0
        x = np.cos(az) * ce
        y = np.sin(az) * ce
        z = np.full_like(az, se)
        dirs[r, :, 0] = x
        dirs[r, :, 1] = y
        dirs[r, :, 2] = z
    return {
        "directions": dirs,
        "ring": np.arange(rings, dtype=np.int16),
        "azimuth": az.astype(np.float32),
    }


# ----------------- public API: raycasting ------------------


@dataclass
class RaycastResult:
    xyz: np.ndarray
    intensity_u8: np.ndarray
    ring: np.ndarray
    azimuth: np.ndarray
    elevation: np.ndarray
    return_id: np.ndarray
    num_returns: np.ndarray
    range_m: np.ndarray
    cos_incidence: Optional[np.ndarray] = None
    mat_class: Optional[np.ndarray] = None
    reflectivity: Optional[np.ndarray] = None
    transmittance: Optional[np.ndarray] = None


def perform_raycasting(
    scene,
    depsgraph,
    origins: np.ndarray,  # (N,3) world
    directions: np.ndarray,  # (N,3) world unit
    rings: np.ndarray,  # (N,)
    azimuth_rad: np.ndarray,  # (N,)
    cfg,
) -> Dict[str, np.ndarray]:
    """
    Cast rays, compute radiometry per hit, optionally spawn one secondary.
    Alpha is applied once here to both reflectivity and intensity.

    Returns dict of numpy arrays suitable for PLY writing.
    """
    assert bpy is not None, "perform_raycasting requires Blender (bpy)"

    min_r = float(getattr(cfg, "min_range", 0.05))
    max_r = float(getattr(cfg, "max_range", 100.0))

    # Secondary settings
    enable_secondary = bool(getattr(cfg, "enable_secondary", False))
    sec_min_res = float(getattr(cfg, "secondary_min_residual", 0.02))
    sec_bias = float(
        getattr(cfg, "secondary_ray_bias", getattr(cfg, "hit_offset", 5e-4))
    )
    sec_min_cos = float(getattr(cfg, "secondary_min_cos", 0.95))
    merge_eps = float(getattr(cfg, "secondary_merge_eps", 0.0))

    # Angle dropout
    grazing_drop = float(getattr(cfg, "grazing_dropout_cos_thresh", 0.05))

    # Output buffers (grow as needed)
    pts, inten_raw, refl_f, rings_out, az_out, elev_out = [], [], [], [], [], []
    ret_id, num_ret, ranges, cos_i_list, mat_cls, trans_list = [], [], [], [], [], []

    # Helper: try a secondary pass-through and return a dict or None
    def _secondary_hit(loc, nrm, d, r, rings_i, az_i):
        o2 = loc + nrm * max(1e-5, sec_bias)
        d2 = d
        hit2, loc2, normal2, face_index2, obj2, _ = scene.ray_cast(
            depsgraph, tuple(o2), tuple(d2), distance=(max_r - r)
        )
        if not hit2:
            return None
        loc2 = np.array(loc2, dtype=np.float64)
        r2 = r + float(np.linalg.norm(loc2 - o2))
        if not (min_r <= r2 <= max_r):
            return None
        nrm2 = _unit(np.array(normal2, dtype=np.float64))
        cos_i2 = _compute_cos_i(nrm2, d2)
        props2 = extract_material_properties(obj2, int(face_index2), depsgraph, loc2)
        I0_2, _, refl0_2, T2, alpha2 = compute_intensity(props2, cos_i2, r2, cfg)
        return {
            "P": loc2.astype(np.float32),
            "I0": float(I0_2),
            "refl0": float(refl0_2),
            "T": float(T2),
            "alpha": float(alpha2),
            "r": r2,
            "cos_i": cos_i2,
            "ring": int(rings_i),
            "az": float(az_i),
            "mat_class": int(classify_material(props2)),
        }

    # Cast loop
    N = int(origins.shape[0])
    for i in range(N):
        o = origins[i].astype(np.float64)
        d = _unit(directions[i].astype(np.float64))

        hit, loc, normal, face_index, obj, _ = scene.ray_cast(
            depsgraph, tuple(o), tuple(d), distance=max_r
        )
        if not hit:
            continue

        loc = np.array(loc, dtype=np.float64)
        nrm = _unit(np.array(normal, dtype=np.float64))
        r = float(np.linalg.norm(loc - o))
        if r < min_r or r > max_r:
            continue

        # Material extraction
        props = extract_material_properties(obj, int(face_index), depsgraph, loc, cfg)
        # Geometric normal only; flip if backfacing
        sh_nrm = nrm
        if np.dot(sh_nrm, d) > 0:
            sh_nrm = -sh_nrm
        cos_i = _compute_cos_i(sh_nrm, d)
        if cos_i < grazing_drop:
            continue

        # Radiometry (pre-alpha reflectivity)
        I0, sec_scale, refl0, T_mat, alpha_cov = compute_intensity(props, cos_i, r, cfg)

        # Alpha handling:
        # - CLIP: cull when coverage below threshold; otherwise do not scale
        # - BLEND/HASHED: never cull by threshold; scale by coverage
        alpha_mode = str(props.get("alpha_mode", "BLEND")).upper()
        alpha_clip = float(props.get("alpha_clip", 0.5))
        if alpha_mode == "CLIP" and alpha_cov < alpha_clip:
            continue
        alpha_apply = 1.0 if alpha_mode == "CLIP" else alpha_cov

        # Apply alpha once
        refl = float(refl0 * alpha_apply)
        I = float(I0 * alpha_apply)

        # Primary record
        pts.append(loc.astype(np.float32))
        inten_raw.append(I)
        refl_f.append(refl)
        rings_out.append(int(rings[i]))
        az_out.append(float(azimuth_rad[i]))
        # approximate elevation from direction
        elev_out.append(float(math.asin(max(-1.0, min(1.0, d[2])))))
        ret_id.append(1)
        ranges.append(r)
        cos_i_list.append(cos_i)
        mat_cls.append(int(classify_material(props)))
        trans_list.append(float(T_mat))

        # Secondary path
        sec_added = False
        if enable_secondary:
            residual = float(
                sec_scale * alpha_apply
            )  # spawn energy from primary surface
            if (
                residual > sec_min_res
                and cos_i >= sec_min_cos
                and (max_r - r) > sec_bias
            ):
                sec = _secondary_hit(
                    loc,
                    nrm if np.isfinite(nrm).all() else d,
                    d,
                    r,
                    rings[i],
                    azimuth_rad[i],
                )
                if sec is not None:
                    eff = residual * sec["alpha"]
                    I2 = float(sec["I0"]) * eff
                    refl2 = float(sec["refl0"]) * eff
                    r2 = sec["r"]
                    if merge_eps > 0.0 and abs(r2 - r) <= merge_eps:
                        if I2 > I:
                            pts[-1] = sec["P"]
                            inten_raw[-1] = I2
                            refl_f[-1] = refl2
                            ranges[-1] = r2
                            cos_i_list[-1] = sec["cos_i"]
                            mat_cls[-1] = sec["mat_class"]
                            trans_list[-1] = sec["T"]
                        sec_added = False
                    else:
                        pts.append(sec["P"])
                        inten_raw.append(I2)
                        refl_f.append(refl2)
                        rings_out.append(sec["ring"])
                        az_out.append(sec["az"])
                        elev_out.append(float(math.asin(max(-1.0, min(1.0, d[2])))))
                        ret_id.append(2)
                        ranges.append(r2)
                        cos_i_list.append(sec["cos_i"])
                        mat_cls.append(sec["mat_class"])
                        trans_list.append(sec["T"])
                        sec_added = True

        # Set num_returns for this beam
        if sec_added:
            num_ret.append(2)
            num_ret.append(2)  # both entries carry total count
        else:
            num_ret.append(1)

    if not pts:
        # Empty outputs with correct dtypes
        return {
            "points": np.zeros((0, 3), np.float32),
            "intensity": np.zeros((0,), np.uint8),
            "ring": np.zeros((0,), np.uint16),
            "azimuth": np.zeros((0,), np.float32),
            "elevation": np.zeros((0,), np.float32),
            "return_id": np.zeros((0,), np.uint8),
            "num_returns": np.zeros((0,), np.uint8),
            "range_m": np.zeros((0,), np.float32),
            "cos_incidence": np.zeros((0,), np.float32),
            "mat_class": np.zeros((0,), np.uint8),
            "reflectivity": np.zeros((0,), np.float32),
            "transmittance": np.zeros((0,), np.float32),
        }

    pts = np.stack(pts, axis=0)
    inten_raw = np.asarray(inten_raw, dtype=np.float32)
    refl_f = _clip01(np.asarray(refl_f, dtype=np.float32))
    rings_out = np.asarray(rings_out, dtype=np.uint16)
    az_out = np.asarray(az_out, dtype=np.float32)
    elev_out = np.asarray(elev_out, dtype=np.float32)
    ret_id = np.asarray(ret_id, dtype=np.uint8)
    num_ret = np.asarray(num_ret, dtype=np.uint8)
    ranges = np.asarray(ranges, dtype=np.float32)
    cos_i_arr = np.asarray(cos_i_list, dtype=np.float32)
    mat_cls = np.asarray(mat_cls, dtype=np.uint8)
    trans_arr = _clip01(np.asarray(trans_list, dtype=np.float32))

    # Exposure mapping to U8
    auto = bool(getattr(cfg, "auto_expose", False))
    global_scale = float(getattr(cfg, "global_scale", 1.0))
    target_pct = float(getattr(cfg, "target_percentile", 95.0))
    target_u8 = float(getattr(cfg, "target_intensity", 200.0))

    pos_mask = inten_raw > 0.0
    if auto and np.count_nonzero(pos_mask) >= 4:
        scale = _percentile_scale(inten_raw[pos_mask], target_pct, target_u8)
    else:
        scale = global_scale
    inten_u8 = np.clip(np.round(inten_raw * scale * 255.0), 0, 255).astype(np.uint8)

    return {
        "points": pts,
        "intensity": inten_u8,
        "ring": rings_out,
        "azimuth": az_out,
        "elevation": elev_out,
        "return_id": ret_id,
        "num_returns": num_ret,
        "range_m": ranges,
        "cos_incidence": cos_i_arr,
        "mat_class": mat_cls,
        "reflectivity": refl_f,
        "transmittance": trans_arr,
        "scale_used": np.float32(scale),
    }
