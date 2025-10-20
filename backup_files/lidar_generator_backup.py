#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Infinigen LiDAR Ground Truth (ray-traced)
- Shader-aware intensity (diffuse/specular/transmissive via Principled BSDF params)
- Multi-echo (optional) with residual transmittance
- Sensor presets (VLP-16, HDL-32E, HDL-64E, OS1-128)
- Rolling shutter + continuous spin semantics
- Consistent per-scan auto-exposure; optional binary PLY; PLY frame selectable
- Performance: vectorized directions, material param caching, evaluated depsgraph

New in this version:
- Emit per-hit incidence cosine (cos_incidence) for angle-aware debugging.
- Emit coarse material class (mat_class: 0=diffuse, 1=transmissive, 2=metallic).
- PLY headers/records include these only when --emit-aux is enabled.
- Fixed binary PLY struct to exactly match declared header fields.

Usage (inside Blender):
  blender -b scene.blend --python lidar_generator.py -- --output_dir out --preset VLP-16 --frames 1-48
"""

import argparse
import bpy
import json
import math
import numpy as np
import os
import random
import sys
import time
import struct
from mathutils import Vector, Matrix

# -------------------------
# LiDAR PRESETS
# -------------------------

LIDAR_PRESETS = {
    "VLP-16": {
        "num_elevation": 16,
        "elevation_angles_deg": [
            -15.0,  1.0,  -13.0,  3.0,  -11.0,  5.0,  -9.0,   7.0,
             -7.0,  9.0,   -5.0, 11.0,  -3.0, 13.0,  -1.0,  15.0
        ]
    },
    "HDL-32E": {
        "num_elevation": 32,
        "elevation_angles_deg": [
            -30.67, -9.33, -29.33, -8.00, -28.00, -6.67, -26.67, -5.33,
            -25.33, -4.00, -24.00, -2.67, -22.67, -1.33, -21.33,  0.00,
            -20.00,  1.33, -18.67,  2.67, -17.33,  4.00, -16.00,  5.33,
            -14.67,  6.67, -13.33,  8.00, -12.00,  9.33, -10.67, 10.67
        ]
    },
    "HDL-64E": {
        "num_elevation": 64,
        "elevation_angles_deg": [
            -24.33, -23.67, -23.0, -22.33, -21.67, -21.0, -20.33, -19.67,
            -19.0, -18.33, -17.67, -17.0, -16.33, -15.67, -15.0, -14.33,
            -13.67, -13.0, -12.33, -11.67, -11.0, -10.33, -9.67, -9.0,
            -8.33, -7.67, -7.0, -6.33, -5.67, -5.0, -4.33, -3.67,
            -3.0, -2.33, -1.67, -1.0, -0.33, 0.33, 1.0, 1.67, 2.33, 3.0,
            3.67, 4.33, 5.0, 5.67, 6.33, 7.0, 7.67, 8.33, 9.0, 9.67,
            10.33, 11.0, 11.67, 12.33, 13.0, 13.67, 14.33, 15.0, 15.67,
            16.33, 17.0, 17.67
        ]
    },
    "OS1-128": {
        "num_elevation": 128,
        "elevation_angles_deg": list(np.linspace(22.5, -22.5, 128))
    }
}

# -------------------------
# Configuration object
# -------------------------

class LidarConfig:
    def __init__(self,
                 preset: str = "VLP-16",
                 num_azimuth: int = 1800,
                 indoor_mode: bool = False,
                 save_ply: bool = True,
                 save_kitti: bool = False,
                 auto_expose: bool = True,
                 global_scale: float = 1.0,
                 emit_aux_fields: bool = True,
                 rpm: float = 600.0,  # 10 Hz spin
                 continuous_spin: bool = True,
                 rolling_shutter: bool = True,
                 multi_echo: bool = False,
                 beta_atm: float = 0.0,
                 ply_binary: bool = False,
                 ply_frame: str = "camera",  # {camera,sensor,world}
                 kitti_intensity_mode: str = "scaled"  # {scaled,reflectance}
                 ):

        if preset not in LIDAR_PRESETS:
            raise ValueError(f"Unknown LiDAR preset: {preset}. Available: {list(LIDAR_PRESETS.keys())}")

        # Geometry
        self.preset = preset
        preset_data = LIDAR_PRESETS[preset]
        self.num_elevation = preset_data["num_elevation"]
        self.elevation_angles_deg = preset_data["elevation_angles_deg"]
        self.num_azimuth = num_azimuth

        # Ranges
        self.min_range = 0.1 if indoor_mode else 0.9
        self.max_range = 100.0

        # Intensity model (simple, interpretable factors)
        self.distance_power = 2.0          # ~1/R^2 default; can soften/tighten
        self.k_cos = 1.2                   # footprint/incidence exponent
        self.target_percentile = 90        # auto-exposure percentile
        self.target_intensity = 192        # percentile -> this U8 value
        self.auto_expose = auto_expose
        self.global_scale = global_scale
        self.beta_atm = beta_atm           # 0 indoors; ~0.004-0.01 outdoors for haze

        # Timing / spin
        self.rpm = rpm
        self.continuous_spin = continuous_spin
        self.rolling_shutter = rolling_shutter

        # Output
        self.save_ply = save_ply
        self.save_kitti = save_kitti
        self.emit_aux_fields = emit_aux_fields
        self.ply_binary = ply_binary
        self.ply_frame = ply_frame
        self.kitti_intensity_mode = kitti_intensity_mode

        # Noise / dropout
        self.range_noise_a = 0.01          # base range noise [m]
        self.range_noise_b = 0.001         # range-proportional noise
        self.intensity_jitter_std = 0.02   # multiplicative intensity jitter
        self.dropout_prob = 0.015          # random dropout
        self.grazing_dropout_cos_thresh = 0.02  # drop ultra-grazing hits

        # Multi-echo
        self.multi_echo = multi_echo

    def to_dict(self):
        return {
            "preset": self.preset,
            "num_elevation": self.num_elevation,
            "num_azimuth": self.num_azimuth,
            "elevation_angles_deg": self.elevation_angles_deg,
            "min_range": self.min_range,
            "max_range": self.max_range,
            "distance_power": self.distance_power,
            "k_cos": self.k_cos,
            "target_percentile": self.target_percentile,
            "target_intensity": self.target_intensity,
            "auto_expose": self.auto_expose,
            "global_scale": self.global_scale,
            "beta_atm": self.beta_atm,
            "rpm": self.rpm,
            "continuous_spin": self.continuous_spin,
            "rolling_shutter": self.rolling_shutter,
            "save_ply": self.save_ply,
            "save_kitti": self.save_kitti,
            "emit_aux_fields": self.emit_aux_fields,
            "ply_binary": self.ply_binary,
            "ply_frame": self.ply_frame,
            "kitti_intensity_mode": self.kitti_intensity_mode,
            "range_noise_a": self.range_noise_a,
            "range_noise_b": self.range_noise_b,
            "intensity_jitter_std": self.intensity_jitter_std,
            "dropout_prob": self.dropout_prob,
            "grazing_dropout_cos_thresh": self.grazing_dropout_cos_thresh,
            "multi_echo": self.multi_echo,
        }

    def __str__(self):
        mode = "indoor" if self.min_range < 0.5 else "spec"
        return f"{self.preset} ({mode}): {self.num_elevation}x{self.num_azimuth}, rpm={self.rpm}, rolling={self.rolling_shutter}, spin={self.continuous_spin}"

# -------------------------
# Scene / camera utils
# -------------------------

def setup_scene(scene_path: str):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.open_mainfile(filepath=scene_path)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.use_persistent_data = True
    return bpy.context.scene

def resolve_camera(name: str | None):
    if name:
        cam = bpy.data.objects.get(name)
        if cam:
            return cam
        print(f"Warning: camera '{name}' not found; using first camera.", file=sys.stderr)
    cams = [o for o in bpy.data.objects if o.type == 'CAMERA']
    if not cams:
        print("Error: no camera in scene.", file=sys.stderr)
        sys.exit(1)
    return cams[0]

def sensor_to_camera_rotation() -> Matrix:
    """Map sensor axes (x fwd, y left, z up) to Blender camera axes (-Z fwd, -X right, +Y up)."""
    return Matrix(((0.0, -1.0,  0.0),
                   (0.0,  0.0,  1.0),
                   (-1.0, 0.0,  0.0)))

# -------------------------
# Rays (sensor frame)
# -------------------------

def generate_sensor_rays(config: LidarConfig):
    """Precompute per-ray metadata for one revolution in the sensor frame (x fwd, y left, z up)."""
    elev = np.array([math.radians(a) for a in config.elevation_angles_deg], dtype=np.float32)
    az = np.linspace(0.0, 2.0 * math.pi, config.num_azimuth, endpoint=False, dtype=np.float32)

    ce = np.cos(elev)
    se = np.sin(elev)

    dirs_zero_yaw, ring_ids, az_idx, elev_arr, az_base = [], [], [], [], []
    for r, (c, s) in enumerate(zip(ce, se)):
        for i, a in enumerate(az):
            dirs_zero_yaw.append((c, 0.0, s))   # +X with elevation; yaw will be applied later
            ring_ids.append(r)
            az_idx.append(i)
            elev_arr.append(float(elev[r]))
            az_base.append(float(a))

    return (
        np.array(dirs_zero_yaw, dtype=np.float32),
        np.array(ring_ids, dtype=np.uint16),
        np.array(az_idx, dtype=np.int32),
        np.array(elev_arr, dtype=np.float32),
        np.array(az_base, dtype=np.float32),
    )

# -------------------------
# Material extraction & caching (Principled BSDF)
# -------------------------

_MATERIAL_CACHE = {}

def _find_principled_bsdf(mat: bpy.types.Material):
    if not mat or not mat.use_nodes or not mat.node_tree:
        return None
    for n in mat.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            return n
    return None

def _safe_input(node, name, default):
    try:
        sock = node.inputs.get(name)
        if sock is None or sock.is_linked:
            return default
        v = sock.default_value
        if hasattr(v, '__len__'):
            if len(v) >= 3:
                return tuple(float(x) for x in v[:3])  # RGB(A) -> RGB
            return float(v[0])
        return float(v)
    except Exception:
        return default

def get_material_from_hit(obj: bpy.types.Object, poly_index: int, depsgraph) -> bpy.types.Material | None:
    """Use evaluated object (after modifiers) to fetch the material used by the hit polygon."""
    try:
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.data
        if not hasattr(mesh, 'polygons') or poly_index < 0:
            return None
        poly = mesh.polygons[poly_index]
        mats = mesh.materials
        if mats and poly.material_index < len(mats):
            return mats[poly.material_index]
        if obj.material_slots:
            return obj.material_slots[poly.material_index].material
    except Exception:
        pass
    return None

def extract_material_properties(obj, poly_index, depsgraph):
    """Return a dict of Principled-like params with caching by material id."""
    mat = get_material_from_hit(obj, poly_index, depsgraph)
    defaults = {
        'base_color': (0.8, 0.8, 0.8),
        'metallic': 0.0,
        'specular': 0.5,
        'roughness': 0.5,
        'transmission': 0.0,
        'ior': 1.45,
        'alpha': 1.0,
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
        params['base_color']   = _safe_input(node, 'Base Color',  params['base_color'])
        params['metallic']     = float(_safe_input(node, 'Metallic',     params['metallic']))
        params['specular']     = float(_safe_input(node, 'Specular',     params['specular']))
        params['roughness']    = float(_safe_input(node, 'Roughness',    params['roughness']))
        params['transmission'] = float(_safe_input(node, 'Transmission', params['transmission']))
        try:
            params['ior'] = float(_safe_input(node, 'IOR', params['ior']))
        except Exception:
            params['ior'] = 1.45
        try:
            params['alpha'] = float(_safe_input(node, 'Alpha', params['alpha']))
        except Exception:
            params['alpha'] = 1.0
    elif hasattr(mat, 'diffuse_color'):
        c = mat.diffuse_color
        params['base_color'] = (float(c[0]), float(c[1]), float(c[2]))

    _MATERIAL_CACHE[key] = params
    return params

# -------------------------
# Intensity model helpers
# -------------------------

def _luma(rgb):
    r, g, b = rgb
    return max(0.0, min(1.0, 0.2126*r + 0.7152*g + 0.0722*b))

def schlick_fresnel(cos_theta, F0):
    m = max(0.0, min(1.0, cos_theta))
    return F0 + (1.0 - F0) * (1.0 - m) ** 5

def f0_from_params(metallic, specular, ior, base_color):
    # Metals use colored F0 ~= base color; dielectrics use scalar F0 from IOR/specular
    if metallic >= 0.5:
        F0 = np.clip(np.array(base_color[:3], dtype=np.float32), 0.0, 1.0)
    else:
        if ior and ior > 0:
            f0 = ((ior - 1.0) / (ior + 1.0)) ** 2
        else:
            f0 = 0.08 * float(specular)
        F0 = np.array([f0, f0, f0], dtype=np.float32)
    return F0

def ggx_ndf(cos_theta, roughness):
    # Trowbridge-Reitz GGX NDF (monostatic backscatter simplification)
    alpha = max(1e-4, float(roughness)) ** 2
    c2 = max(1e-6, cos_theta) ** 2
    denom = math.pi * (c2 * (alpha - 1.0) + 1.0) ** 2
    return alpha / max(1e-6, denom)

def spec_backscatter(cos_theta, roughness, F0_rgb):
    # Luminance of specular lobe ~ Fresnel * NDF (geometry term omitted)
    D = ggx_ndf(cos_theta, roughness)
    F = schlick_fresnel(cos_theta, F0_rgb)  # vectorized per RGB
    return _luma(F * D)

def transmissive_reflectance(cos_theta, ior):
    # Air/dielectric Fresnel reflectance for first surface (no multiple scattering)
    f0 = ((ior - 1.0) / (ior + 1.0)) ** 2 if ior and ior > 0 else 0.04
    return schlick_fresnel(cos_theta, f0)

def compute_intensity(props: dict, cos_i: float, R: float, cfg: LidarConfig, indoor: bool):
    """
    Compute plausible LiDAR return (pre-exposure).
    Mix diffuse/specular/transmissive terms, then apply:
      - cos^k (beam footprint + extraction effects)
      - 1/R^p (range attenuation)
      - e^{-beta * R} (optional atmosphere)
    Returns (I_raw, residual_transmittance_for_second_echo).
    """
    base_rgb = np.clip(np.array(props['base_color'][:3], dtype=np.float32), 0.0, 1.0)
    alb   = _luma(base_rgb)
    rough = float(props['roughness'])
    metal = float(props['metallic'])
    spec  = float(props['specular'])
    trans = float(props.get('transmission', 0.0))
    ior   = float(props.get('ior', 1.45)) if props.get('ior', None) is not None else 1.45

    # Heuristic weights: metals -> specular; transmissive / low alpha -> transmissive
    w_s = 0.15 + 0.75 * metal
    alpha = float(props.get('alpha', 1.0))
    w_t = max(trans, 1.0 - alpha)
    w_d = max(0.0, 1.0 - w_s - 0.5 * w_t)
    s = max(1e-6, w_d + w_s + w_t)
    w_d, w_s, w_t = w_d / s, w_s / s, w_t / s

    rho_d = alb * cos_i
    F0    = f0_from_params(metal, spec, ior, base_rgb)
    rho_s = spec_backscatter(cos_i, rough, F0)
    rho_t = transmissive_reflectance(cos_i, ior) if w_t > 1e-3 else 0.0

    cos_term = cos_i ** cfg.k_cos
    att_rng  = 1.0 / max(1e-3, R ** cfg.distance_power)
    att_atm  = math.exp(-cfg.beta_atm * R) if cfg.beta_atm > 0 else 1.0

    rho_mix = w_d * rho_d + w_s * rho_s + w_t * rho_t
    I = rho_mix * cos_term * att_rng * att_atm
    if indoor:
        I *= 1.1  # mild boost indoors to fill in darker scenes

    residual_T = max(0.0, 1.0 - rho_t) if w_t > 0.15 else 1.0
    return float(I), residual_T

# -------------------------
# Ray casting & scan assembly
# -------------------------

def _classify_material(props: dict) -> int:
    """Map Principled params to a coarse class: 0=diffuse, 1=transmissive, 2=metallic."""
    metal = float(props.get('metallic', 0.0))
    trans = float(props.get('transmission', 0.0))
    alpha = float(props.get('alpha', 1.0))
    if metal >= 0.5:
        return 2
    if max(trans, 1.0 - alpha) >= 0.3:
        return 1
    return 0

def perform_raycasting(scene, depsgraph, origin_world, world_dirs, ring_ids, az_rad, el_rad, t_offsets, cfg: LidarConfig):
    points_world, rings = [], []
    intens_raw, refl_raw = [], []
    az_hit, el_hit, t_hit = [], [], []
    ret_id, num_rets = [], []

    # NEW: debug/analysis fields
    cos_list = []         # incidence cosine
    mat_class_list = []   # coarse material class

    for i, d in enumerate(world_dirs):
        if random.random() < cfg.dropout_prob:
            continue

        dv = Vector(d)
        hit, loc, nrm, poly_idx, obj, _ = scene.ray_cast(depsgraph, origin_world, dv, distance=cfg.max_range)
        if not hit or not obj:
            continue

        dist = (loc - origin_world).length
        if not (cfg.min_range <= dist <= cfg.max_range):
            continue

        n = Vector(nrm).normalized()
        # Incidence cosine: n · (-d). Clamp negatives to 0 (backfaces ignored)
        cos_i = max(0.0, float(n.dot(-dv)))
        if cos_i < cfg.grazing_dropout_cos_thresh:
            continue

        props = extract_material_properties(obj, poly_idx, depsgraph)
        I01, residual_T = compute_intensity(props, cos_i, dist, cfg, indoor=(cfg.min_range < 0.5))

        # Intensity jitter (multiplicative)
        I01 *= max(0.0, 1.0 + random.gauss(0.0, cfg.intensity_jitter_std))

        # Range noise applies to stored point position (not to intensity)
        sigma_r = cfg.range_noise_a + cfg.range_noise_b * dist
        dist_noisy = max(cfg.min_range, dist + random.gauss(0.0, sigma_r))
        loc_noisy = origin_world + dv * dist_noisy

        # Save primary return
        points_world.append(np.array(loc_noisy, dtype=np.float32))
        rings.append(ring_ids[i])
        intens_raw.append(I01)
        refl_raw.append(I01)   # same scalar pre-exposure (kept for clarity)
        az_hit.append(az_rad[i])
        el_hit.append(el_rad[i])
        t_hit.append(t_offsets[i])
        ret_id.append(1)
        num_rets.append(1)

        # NEW: extra debug signals
        cos_list.append(cos_i)
        mat_class_list.append(_classify_material(props))

        # Optional second return through transmissive materials
        if cfg.multi_echo and residual_T < 1.0:
            origin2 = loc + dv * 1e-3  # step past first surface
            hit2, loc2, nrm2, poly_idx2, obj2, _ = scene.ray_cast(depsgraph, origin2, dv, distance=(cfg.max_range - dist))
            if hit2 and obj2:
                dist2 = (loc2 - origin_world).length
                if cfg.min_range <= dist2 <= cfg.max_range:
                    n2 = Vector(nrm2).normalized()
                    cos_i2 = max(0.0, float(n2.dot(-dv)))
                    if cos_i2 >= cfg.grazing_dropout_cos_thresh:
                        props2 = extract_material_properties(obj2, poly_idx2, depsgraph)
                        I02, _ = compute_intensity(props2, cos_i2, dist2, cfg, indoor=(cfg.min_range < 0.5))
                        I02 *= residual_T
                        I02 *= max(0.0, 1.0 + random.gauss(0.0, cfg.intensity_jitter_std))

                        # noisy position for second hit
                        sigma_r2 = cfg.range_noise_a + cfg.range_noise_b * dist2
                        dist2_noisy = max(cfg.min_range, dist2 + random.gauss(0.0, sigma_r2))
                        loc2_noisy = origin_world + dv * dist2_noisy

                        points_world.append(np.array(loc2_noisy, dtype=np.float32))
                        rings.append(ring_ids[i])
                        intens_raw.append(I02)
                        refl_raw.append(I02)
                        az_hit.append(az_rad[i])
                        el_hit.append(el_rad[i])
                        t_hit.append(t_offsets[i] + 1e-6)
                        ret_id.append(2)
                        num_rets[-1] = 2  # mark previous as having 2 returns
                        num_rets.append(2)

                        # NEW: signals for second hit
                        cos_list.append(cos_i2)
                        mat_class_list.append(_classify_material(props2))

    if not points_world:
        return None

    # Auto-exposure: percentile -> target U8; otherwise use global_scale
    raw = np.array(intens_raw, dtype=np.float32)
    if cfg.auto_expose and raw.size >= 4:
        p = np.percentile(raw[raw > 0], cfg.target_percentile) if np.any(raw > 0) else 1.0
        scale = (cfg.target_intensity / 255.0) / max(1e-6, p)
    else:
        scale = float(cfg.global_scale) / 255.0

    ints_u8 = np.clip(np.round(raw * scale * 255.0), 0, 255).astype(np.uint8)

    return {
        "points_world":   np.vstack(points_world),
        "ring_ids":       np.array(rings, dtype=np.uint16),
        "intensities_u8": ints_u8,
        "reflectance_raw": np.array(refl_raw, dtype=np.float32),   # pre-exposure
        "azimuth_rad":    np.array(az_hit, dtype=np.float32),
        "elevation_rad":  np.array(el_hit, dtype=np.float32),
        "time_offset":    np.array(t_hit, dtype=np.float32),
        "return_id":      np.array(ret_id, dtype=np.uint8),
        "num_returns":    np.array(num_rets, dtype=np.uint8),
        "scale_used":     float(scale),

        # NEW fields
        "cos_incidence":  np.array(cos_list, dtype=np.float32),
        "mat_class":      np.array(mat_class_list, dtype=np.uint8),
    }

# -------------------------
# Saving utilities (PLY / KITTI)
# -------------------------

def _matrix_to_np3x3(m: Matrix) -> np.ndarray:
    return np.array([[m[0][0], m[0][1], m[0][2]],
                     [m[1][0], m[1][1], m[1][2]],
                     [m[2][0], m[2][1], m[2][2]]], dtype=np.float32)

def world_to_frame_matrix(cam_obj, sensor_R: Matrix, frame: str) -> Matrix:
    """Return transform that maps world points into the requested output frame."""
    if frame == 'world':
        return Matrix.Identity(4)
    cam_inv = cam_obj.matrix_world.inverted()
    if frame == 'camera':
        return cam_inv
    if frame == 'sensor':
        # world->sensor = (sensor_R^{-1}) * (world->camera)
        R_sc = sensor_R.inverted().to_4x4()
        return R_sc @ cam_inv
    raise ValueError("ply_frame must be one of {'camera','sensor','world'}")

def save_ply(output_dir, frame, xform_world_to_out: Matrix, data, emit_aux: bool, binary: bool):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"lidar_frame_{frame:04d}.ply")

    # Transform to chosen frame
    pts_out = np.array([(xform_world_to_out @ Vector(p))[:] for p in data["points_world"]], dtype=np.float32)

    n = len(pts_out)
    rings = data["ring_ids"]
    intens_u8 = data["intensities_u8"]

    # Optional fields available?
    has_ci = "cos_incidence" in data
    has_mc = "mat_class" in data

    if not binary:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"ply\nformat ascii 1.0\ncomment Lidar frame {frame}\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar intensity\nproperty ushort ring\n")
            if emit_aux:
                f.write("property float azimuth\nproperty float elevation\nproperty float time_offset\n")
                f.write("property uchar return_id\nproperty uchar num_returns\n")
                if has_ci: f.write("property float cos_incidence\n")
                if has_mc: f.write("property uchar mat_class\n")
            f.write("end_header\n")

            az, el, t = data["azimuth_rad"], data["elevation_rad"], data["time_offset"]
            ret, nret = data["return_id"], data["num_returns"]
            ci = data.get("cos_incidence", None)
            mc = data.get("mat_class", None)

            for i in range(n):
                x, y, z = pts_out[i]
                if emit_aux:
                    line = (f"{x:.6f} {y:.6f} {z:.6f} {int(intens_u8[i])} {int(rings[i])} "
                            f"{az[i]:.6f} {el[i]:.6f} {t[i]:.6e} {int(ret[i])} {int(nret[i])}")
                    if has_ci: line += f" {ci[i]:.6f}"
                    if has_mc: line += f" {int(mc[i])}"
                    f.write(line + "\n")
                else:
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(intens_u8[i])} {int(rings[i])}\n")
        return

    # Binary little-endian PLY
    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"comment Lidar frame {frame}\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar intensity\nproperty ushort ring\n"
        )
        if emit_aux:
            header += (
                "property float azimuth\nproperty float elevation\nproperty float time_offset\n"
                "property uchar return_id\nproperty uchar num_returns\n"
            )
            if has_ci:
                header += "property float cos_incidence\n"
            if has_mc:
                header += "property uchar mat_class\n"
        header += "end_header\n"
        f.write(header.encode("ascii"))

        az, el, t = data["azimuth_rad"], data["elevation_rad"], data["time_offset"]
        ret, nret = data["return_id"], data["num_returns"]
        ci = data.get("cos_incidence", None)
        mc = data.get("mat_class", None)

        # Build a struct that exactly matches our header order
        fmt = "<fffBH"  # x,y,z (f) + intensity (B) + ring (H)
        if emit_aux:
            fmt += "fffBB"  # azimuth, elevation, time_offset (f), return_id (B), num_returns (B)
            if has_ci:
                fmt += "f"   # cos_incidence
            if has_mc:
                fmt += "B"   # mat_class

        pack = struct.Struct(fmt).pack

        for i in range(n):
            fields = [float(pts_out[i][0]), float(pts_out[i][1]), float(pts_out[i][2]),
                      int(intens_u8[i]), int(rings[i])]
            if emit_aux:
                fields += [float(az[i]), float(el[i]), float(t[i]), int(ret[i]), int(nret[i])]
                if has_ci: fields.append(float(ci[i]))
                if has_mc: fields.append(int(mc[i]))
            f.write(pack(*fields))

def save_kitti_bin_sensor(output_dir, frame, cam_obj, sensor_R: Matrix, data, cfg: LidarConfig):
    """Save KITTI-style binary [x,y,z,intensity] in SENSOR frame. Intensity in [0,1]."""
    os.makedirs(output_dir, exist_ok=True)
    bin_path = os.path.join(output_dir, f"lidar_frame_{frame:04d}.bin")

    cam_inv = cam_obj.matrix_world.inverted()
    R_sc = sensor_R.inverted().to_4x4()
    world_to_sensor = R_sc @ cam_inv
    pts_sensor = np.array([(world_to_sensor @ Vector(p))[:] for p in data["points_world"]], dtype=np.float32)

    if cfg.kitti_intensity_mode == 'reflectance':
        intens = data['reflectance_raw']
        denom = max(1e-6, np.percentile(intens, 99.5))  # robust normalization
        intens01 = np.clip(intens / denom, 0.0, 1.0).astype(np.float32)
    else:
        intens01 = (data['intensities_u8'].astype(np.float32) / 255.0)

    out = np.concatenate([pts_sensor, intens01.reshape(-1, 1)], axis=1).astype(np.float32)
    out.tofile(bin_path)

# -------------------------
# Per-frame processing
# -------------------------

def process_frame(scene, cam_obj, frame, fps, output_dir, cfg: LidarConfig, sensor_R: Matrix, precomp, phase_offset_rad):
    bpy.context.scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()

    dirs_sensor, ring_ids, az_idx, elev_rad, az_base = precomp

    frame_dt = 1.0 / fps
    omega = 2.0 * math.pi * (cfg.rpm / 60.0)  # rad/s

    # Spin/timing semantics
    if cfg.continuous_spin:
        if cfg.rolling_shutter:
            t_offsets = (az_idx.astype(np.float32) / float(cfg.num_azimuth)) * frame_dt
        else:
            t_offsets = np.zeros_like(az_base)
        az = az_base + phase_offset_rad + (omega * t_offsets)
    else:
        t_offsets = np.zeros_like(az_base)
        az = az_base + phase_offset_rad

    # Yaw the sensor-frame directions (vectorized)
    ca, sa = np.cos(az), np.sin(az)
    dirs_yawed = np.stack([
        dirs_sensor[:, 0] * ca - dirs_sensor[:, 1] * sa,
        dirs_sensor[:, 0] * sa + dirs_sensor[:, 1] * ca,
        dirs_sensor[:, 2]
    ], axis=1)

    # Map sensor -> camera -> world
    R_cam_np = _matrix_to_np3x3(cam_obj.matrix_world.to_3x3())
    sensor_to_cam_np = _matrix_to_np3x3(sensor_R)
    world_dirs_np = dirs_yawed @ sensor_to_cam_np.T @ R_cam_np.T
    world_dirs = world_dirs_np / np.linalg.norm(world_dirs_np, axis=1, keepdims=True)

    origin = cam_obj.matrix_world.translation

    t0 = time.time()
    res = perform_raycasting(scene, depsgraph, origin, world_dirs, ring_ids, az, elev_rad, t_offsets, cfg)
    dt = time.time() - t0

    if not res:
        print(f"Frame {frame}: cast {len(world_dirs)} rays, hit 0 points ({dt:.2f}s)")
        return 0, phase_offset_rad, None

    nhit = len(res["points_world"])
    print(f"Frame {frame}: cast {len(world_dirs)} rays, hit {nhit} points ({dt:.2f}s, scale={res['scale_used']:.5f})")

    # Save outputs
    if cfg.save_ply:
        xform = world_to_frame_matrix(cam_obj, sensor_R, cfg.ply_frame)
        save_ply(output_dir, frame, xform, res, cfg.emit_aux_fields, cfg.ply_binary)
    if cfg.save_kitti:
        save_kitti_bin_sensor(output_dir, frame, cam_obj, sensor_R, res, cfg)

    # Update phase for continuous spin
    if cfg.continuous_spin:
        phase_offset_rad = (phase_offset_rad + omega * frame_dt) % (2.0 * math.pi)

    return nhit, phase_offset_rad, res

# -------------------------
# CLI / main
# -------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(description="Generate ray-traced LiDAR ground truth.")
    p.add_argument("scene_path", help="Path to Blender .blend scene")
    p.add_argument("--output_dir", default="outputs/infinigen_lidar", help="Output directory")
    p.add_argument("--frames", default="1-48", help="Frame range, e.g. '1-48' or '1,5,10'")
    p.add_argument("--camera", default=None, help="Camera object name")
    p.add_argument("--preset", default="VLP-16", choices=LIDAR_PRESETS.keys(), help="LiDAR sensor model preset")
    p.add_argument("--azimuths", type=int, default=1800, help="Horizontal azimuth steps per revolution")

    # Toggles & options
    p.add_argument("--indoor-mode", dest="indoor_mode", action="store_true")
    p.add_argument("--spec-range", dest="indoor_mode", action="store_false")
    p.set_defaults(indoor_mode=False)

    p.add_argument("--save-ply", dest="save_ply", action="store_true")
    p.add_argument("--no-save-ply", dest="save_ply", action="store_false")
    p.set_defaults(save_ply=True)

    p.add_argument("--save-kitti", dest="save_kitti", action="store_true")
    p.add_argument("--no-save-kitti", dest="save_kitti", action="store_false")
    p.set_defaults(save_kitti=False)

    p.add_argument("--auto-expose", dest="auto_expose", action="store_true")
    p.add_argument("--no-auto-expose", dest="auto_expose", action="store_false")
    p.set_defaults(auto_expose=True)

    p.add_argument("--global-scale", type=float, default=1.0, help="If no auto-exposure, multiply raw intensity by this then map to U8")
    p.add_argument("--beta-atm", type=float, default=0.0, help="Atmospheric attenuation coefficient (m^-1); 0 for indoor")

    p.add_argument("--emit-aux", dest="emit_aux", action="store_true")
    p.add_argument("--no-emit-aux", dest="emit_aux", action="store_false")
    p.set_defaults(emit_aux=True)

    p.add_argument("--rpm", type=float, default=600.0, help="Sensor RPM (600=10Hz)")
    p.add_argument("--continuous-spin", dest="continuous_spin", action="store_true")
    p.add_argument("--no-continuous-spin", dest="continuous_spin", action="store_false")
    p.set_defaults(continuous_spin=True)

    p.add_argument("--rolling-shutter", dest="rolling_shutter", action="store_true")
    p.add_argument("--no-rolling-shutter", dest="rolling_shutter", action="store_false")
    p.set_defaults(rolling_shutter=True)

    p.add_argument("--multi-echo", dest="multi_echo", action="store_true")
    p.add_argument("--no-multi-echo", dest="multi_echo", action="store_false")
    p.set_defaults(multi_echo=False)

    p.add_argument("--binary-ply", dest="binary_ply", action="store_true")
    p.add_argument("--ascii-ply", dest="binary_ply", action="store_false")
    p.set_defaults(binary_ply=False)

    p.add_argument("--ply-frame", choices=["camera", "sensor", "world"], default="camera",
                   help="Which frame to write PLY points in")

    p.add_argument("--kitti-intensity-mode", choices=["scaled", "reflectance"], default="scaled",
                   help="Intensity in KITTI .bin: scaled (matches PLY) or reflectance (raw normalized)")

    return p.parse_args(argv)

def parse_frame_list(spec: str):
    if "-" in spec:
        a, b = map(int, spec.split("-"))
        return list(range(a, b + 1))
    return list(map(int, spec.split(",")))

def main():
    script_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parse_args(script_args)

    frames = parse_frame_list(args.frames)
    scene = setup_scene(args.scene_path)
    cam = resolve_camera(args.camera)

    cfg = LidarConfig(
        preset=args.preset,
        num_azimuth=args.azimuths,
        indoor_mode=args.indoor_mode,
        save_ply=args.save_ply,
        save_kitti=args.save_kitti,
        auto_expose=args.auto_expose,
        global_scale=args.global_scale,
        emit_aux_fields=args.emit_aux,
        rpm=args.rpm,
        continuous_spin=args.continuous_spin,
        rolling_shutter=args.rolling_shutter,
        multi_echo=args.multi_echo,
        beta_atm=args.beta_atm,
        ply_binary=args.binary_ply,
        ply_frame=args.ply_frame,
        kitti_intensity_mode=args.kitti_intensity_mode,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "lidar_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    sensor_R = sensor_to_camera_rotation()
    precomp = generate_sensor_rays(cfg)

    total_pts = 0
    phase = 0.0
    metadata = {"frames": {}}
    traj = {}

    fps = scene.render.fps / max(scene.render.fps_base, 1.0)
    t_all = time.time()

    for fr in frames:
        nhit, phase, res = process_frame(scene, cam, fr, fps, args.output_dir, cfg, sensor_R, precomp, phase)
        total_pts += nhit

        Mw = cam.matrix_world
        R = Mw.to_3x3()
        t = Mw.translation
        traj[fr] = {"R": [[R[0][0], R[0][1], R[0][2]],
                           [R[1][0], R[1][1], R[1][2]],
                           [R[2][0], R[2][1], R[2][2]]],
                    "t": [t[0], t[1], t[2]]}

        metadata["frames"][fr] = {"points": int(nhit), "scale_used": float(res["scale_used"]) if res else 1.0}

    with open(os.path.join(args.output_dir, "trajectory.json"), "w", encoding="utf-8") as f:
        json.dump(traj, f, indent=2)
    with open(os.path.join(args.output_dir, "frame_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    dt_all = time.time() - t_all
    per_frame = dt_all / max(1, len(frames))
    print(f"Done\nFrames: {len(frames)}, total points: {total_pts:,}\nTime: {dt_all:.2f}s (avg {per_frame:.2f}s/frame)")
    print(f"Output: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
