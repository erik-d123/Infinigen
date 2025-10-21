#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# LiDAR ray casting module
# Handles ray pattern generation and ray casting with material-aware intensity

import math
import random
import numpy as np
from mathutils import Vector


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


def _compute_shading_normal(obj, depsgraph, poly_index, hit_world, fallback_normal):
    try:
        eval_obj = obj.evaluated_get(depsgraph)
    except ReferenceError:
        return fallback_normal
    mesh = eval_obj.data
    mesh.calc_normals_split()
    mesh.calc_loop_triangles()
    inv_world = eval_obj.matrix_world.inverted()
    hit_local = inv_world @ hit_world
    for tri in mesh.loop_triangles:
        if tri.polygon_index != poly_index:
            continue
        verts = [mesh.vertices[i].co for i in tri.vertices]
        bary = _barycentric_coords(verts, hit_local)
        if bary is None:
            continue
        loops = tri.loops
        normal_local = Vector((0.0, 0.0, 0.0))
        for weight, loop_index in zip(bary, loops):
            normal_local += weight * mesh.loops[loop_index].normal
        normal_world = eval_obj.matrix_world.to_3x3() @ normal_local
        if normal_world.length_squared > 1e-12:
            return normal_world.normalized()
    return fallback_normal
# Support both package and script execution contexts
try:
    from .intensity_model import extract_material_properties, compute_intensity, classify_material
except Exception:
    from intensity_model import extract_material_properties, compute_intensity, classify_material

def _beer_lambert_transmittance(obj, depsgraph, entry_loc, direction_world, bias, extinction_coeff):
    if extinction_coeff <= 0.0:
        return 1.0
    try:
        eval_obj = obj.evaluated_get(depsgraph)
    except ReferenceError:
        return 1.0
    mat_inv = eval_obj.matrix_world.inverted()
    origin_world = entry_loc + direction_world * max(bias, 1e-4)
    origin_local = mat_inv @ origin_world
    dir_local = mat_inv.to_3x3() @ direction_world
    if dir_local.length_squared == 0.0:
        return 1.0
    dir_local.normalize()
    try:
        hit, location_local, *_ = eval_obj.ray_cast(origin_local, dir_local)
    except AttributeError:
        return 1.0
    if not hit:
        return 1.0
    exit_world = eval_obj.matrix_world @ location_local
    thickness = max(0.0, (exit_world - entry_loc).length)
    return math.exp(-extinction_coeff * thickness)


def generate_sensor_rays(config):
    # Generate ray directions for LiDAR sensor pattern
    elev = np.array([math.radians(a) for a in config.elevation_angles_deg], dtype=np.float32)
    az = np.linspace(0.0, 2.0 * math.pi, config.num_azimuth, endpoint=False, dtype=np.float32)

    ce = np.cos(elev)
    se = np.sin(elev)

    # Build ray directions and metadata
    dirs_zero_yaw, ring_ids, az_idx, elev_arr, az_base = [], [], [], [], []
    for r, (c, s) in enumerate(zip(ce, se)):
        for i, a in enumerate(az):
            dirs_zero_yaw.append((c, 0.0, s))   # +X with elevation; yaw applied later
            ring_ids.append(r)
            az_idx.append(i)
            elev_arr.append(float(elev[r]))
            az_base.append(float(a))

    return (
        np.array(dirs_zero_yaw, dtype=np.float32),
        np.array(ring_ids, dtype=np.uint16),
        np.array(az_idx, dtype=np.uint16),
        np.array(elev_arr, dtype=np.float32),
        np.array(az_base, dtype=np.float32)
    )

def perform_raycasting(scene, depsgraph, origin_world, world_dirs, ring_ids, az_rad, el_rad, t_offsets, cfg):
    # Cast rays and compute LiDAR returns with material-aware intensities
    
    # Output collections
    points_world, rings = [], []
    intens_raw, return_power = [], []
    ranges_m = []
    exposure_scales = []
    az_hit, el_hit, t_hit = [], [], []
    ret_id, num_rets = [], []
    cos_list, mat_class_list = [], []  # Debug/analysis fields
    normals_world = []
    base_eps = max(1e-4, 1e-5 * max(cfg.min_range, 1.0))  # offset to avoid immediate self-intersections

    for i, d in enumerate(world_dirs):
        # Random dropout simulation
        if random.random() < cfg.dropout_prob:
            continue
        dv = Vector(d)
        origin_eps = origin_world + dv * base_eps
        hit, loc, nrm, poly_idx, obj, _ = scene.ray_cast(
            depsgraph, origin_eps, dv, distance=cfg.max_range
        )

        if not hit or not obj:
            continue

        returns = []

        def append_return(point_vec, intensity_val, cos_val, normal_vec, mat_cls, range_val, return_index, total_returns, exposure_scale):
            points_world.append(np.array(point_vec, dtype=np.float32))
            rings.append(ring_ids[i])
            intens_raw.append(intensity_val)
            return_power.append(intensity_val)
            ranges_m.append(range_val)
            az_hit.append(az_rad[i])
            el_hit.append(el_rad[i])
            t_hit.append(t_offsets[i])
            ret_id.append(return_index)
            num_rets.append(total_returns)
            cos_list.append(cos_val)
            mat_class_list.append(mat_cls)
            normals_world.append(np.array(normal_vec, dtype=np.float32))
            exposure_scales.append(exposure_scale)

        dist = (loc - origin_world).length
        if cfg.min_range <= dist <= cfg.max_range:
            face_normal = Vector(nrm).normalized()
            n = _compute_shading_normal(obj, depsgraph, poly_idx, loc, face_normal)
            cos_i = max(0.0, float(n.dot(-dv)))
            if cos_i >= cfg.grazing_dropout_cos_thresh:
                props = extract_material_properties(obj, poly_idx, depsgraph, loc)
                I01, residual_T = compute_intensity(props, cos_i, dist, cfg)
                if I01 > 0.0:
                    I01 *= max(0.0, 1.0 + random.gauss(0.0, cfg.intensity_jitter_std))
                    sigma_floor = cfg.range_noise_a + cfg.range_noise_b * dist
                    sigma_r = sigma_floor + 0.02 / math.sqrt(max(I01, 1e-6))
                    dist_noisy = max(cfg.min_range, dist + random.gauss(0.0, sigma_r))
                    loc_noisy = origin_world + dv * dist_noisy
                    exposure_scale_primary = 1.0
                    returns.append(
                        {
                            "point": loc_noisy,
                            "intensity": I01,
                            "cos": cos_i,
                            "normal": n,
                            "mat_class": classify_material(props),
                            "range": dist_noisy,
                            "exposure": exposure_scale_primary,
                        }
                    )
                residual_T = max(0.0, residual_T)

                if (
                    cfg.enable_secondary
                    and residual_T > cfg.secondary_min_residual
                    and (cfg.max_range - dist) > cfg.secondary_ray_bias
                    and cos_i >= getattr(cfg, "secondary_min_cos", 0.95)
                ):
                    origin_second = loc + dv * cfg.secondary_ray_bias
                    remaining_dist = max(cfg.max_range - dist, cfg.secondary_ray_bias)
                    hit2, loc2, nrm2, poly_idx2, obj2, _ = scene.ray_cast(
                        depsgraph, origin_second, dv, distance=remaining_dist
                    )
                    if hit2 and obj2:
                        total_dist = (loc2 - origin_world).length
                        if cfg.min_range <= total_dist <= cfg.max_range:
                            face_normal2 = Vector(nrm2).normalized()
                            n2 = _compute_shading_normal(obj2, depsgraph, poly_idx2, loc2, face_normal2)
                            cos_i2 = max(0.0, float(n2.dot(-dv)))
                            if cos_i2 >= cfg.grazing_dropout_cos_thresh:
                                props2 = extract_material_properties(obj2, poly_idx2, depsgraph, loc2)
                                I02, _ = compute_intensity(props2, cos_i2, total_dist, cfg)

                                transmission_scale = residual_T
                                if cfg.secondary_extinction > 0.0:
                                    transmission_scale *= _beer_lambert_transmittance(
                                        obj, depsgraph, loc, dv, cfg.secondary_ray_bias, cfg.secondary_extinction
                                    )
                                F_exit = transmissive_reflectance(cos_i2, props2.get("ior", 1.45))
                                transmission_scale *= max(0.0, 1.0 - F_exit)

                                I02 *= transmission_scale
                                if I02 > 0.0:
                                    I02 *= max(0.0, 1.0 + random.gauss(0.0, cfg.intensity_jitter_std))
                                    sigma_floor2 = cfg.range_noise_a + cfg.range_noise_b * total_dist
                                    sigma_r2 = sigma_floor2 + 0.02 / math.sqrt(max(I02, 1e-6))
                                    dist_noisy2 = max(cfg.min_range, total_dist + random.gauss(0.0, sigma_r2))
                                    loc_noisy2 = origin_world + dv * dist_noisy2
                                    returns.append(
                                        {
                                            "point": loc_noisy2,
                                            "intensity": I02,
                                            "cos": cos_i2,
                                            "normal": n2,
                                            "mat_class": classify_material(props2),
                                            "range": dist_noisy2,
                                            "exposure": transmission_scale,
                                        }
                                    )

        if not returns:
            continue

        returns.sort(key=lambda r: r["range"])
        total_returns = len(returns)
        for idx_ret, ret in enumerate(returns, start=1):
            append_return(
                ret["point"],
                ret["intensity"],
                ret["cos"],
                ret["normal"],
                ret["mat_class"],
                ret["range"],
                idx_ret,
                total_returns,
                ret["exposure"],
            )

    if not points_world:
        return None

    # Auto-exposure: map percentile to target intensity
    raw = np.array(intens_raw, dtype=np.float32)
    if cfg.auto_expose and raw.size >= 4:
        positive = raw[raw > 0]
        p = np.percentile(positive, cfg.target_percentile) if positive.size else 1.0
        scale = (cfg.target_intensity / 255.0) / max(1e-6, p)
    else:
        scale = float(cfg.global_scale) / 255.0

    ints_u8 = np.clip(np.round(raw * scale * 255.0), 0, 255).astype(np.uint8)

    # Return structured data
    return {
        "points_world": np.vstack(points_world),
        "ring_ids": np.array(rings, dtype=np.uint16),
        "intensities_u8": ints_u8,
        "return_power": np.array(return_power, dtype=np.float32),
        "range_m": np.array(ranges_m, dtype=np.float32),
        "azimuth_rad": np.array(az_hit, dtype=np.float32),
        "elevation_rad": np.array(el_hit, dtype=np.float32),
        "time_offset": np.array(t_hit, dtype=np.float32),
        "return_id": np.array(ret_id, dtype=np.uint8),
        "num_returns": np.array(num_rets, dtype=np.uint8),
        "scale_used": float(scale),
        "cos_incidence": np.array(cos_list, dtype=np.float32),
        "mat_class": np.array(mat_class_list, dtype=np.uint8),
        "normals_world": np.vstack(normals_world),
        "exposure_scale": np.array(exposure_scales, dtype=np.float32),
    }
