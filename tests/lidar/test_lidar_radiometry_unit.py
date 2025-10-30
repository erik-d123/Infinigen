#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np

from lidar.lidar_config import LidarConfig
from lidar.intensity_model import compute_intensity


def _base_props(**over):
    p = dict(
        base_color=(0.6, 0.6, 0.6),
        metallic=0.0,
        specular=0.5,
        roughness=0.2,
        transmission=0.0,
        transmission_roughness=0.0,
    )
    p.update(over)
    return p


def test_default_opacity_fallback():
    cfg = LidarConfig()
    cfg.default_opacity = 0.3
    props = _base_props()
    _, _, _, _, alpha_cov = compute_intensity(props, cos_i=1.0, R=3.0, cfg=cfg)
    assert abs(alpha_cov - 0.3) < 1e-6


def test_angle_decreases_intensity():
    cfg = LidarConfig()
    props = _base_props()
    I0, *_ = compute_intensity(props, cos_i=1.0, R=5.0, cfg=cfg)
    I1, *_ = compute_intensity(props, cos_i=math.cos(math.radians(40.0)), R=5.0, cfg=cfg)
    assert I1 < I0


def test_distance_falloff():
    cfg = LidarConfig()
    props = _base_props()
    cfg.distance_power = 2.0
    I2, *_ = compute_intensity(props, cos_i=1.0, R=2.0, cfg=cfg)
    I4, *_ = compute_intensity(props, cos_i=1.0, R=4.0, cfg=cfg)
    # Expect roughly 4x drop when doubling distance with p=2
    assert I2 > 0 and I4 > 0 and (I2 / I4) > 3.5


def test_transmission_reduces_reflectivity_and_residual_positive():
    cfg = LidarConfig()
    opaque = _base_props(transmission=0.0)
    glassy = _base_props(transmission=0.85)
    _, res0, refl0, T0, _ = compute_intensity(opaque, cos_i=0.95, R=3.0, cfg=cfg)
    _, res1, refl1, T1, _ = compute_intensity(glassy, cos_i=0.95, R=3.0, cfg=cfg)
    assert refl1 < refl0
    assert T1 > T0
    assert res1 > 0.0


def test_metallic_mixing_increases_fresnel():
    cfg = LidarConfig()
    # bright base, small specular ensures metallic case dominates F0
    dielec = _base_props(base_color=(0.9, 0.9, 0.9), metallic=0.0, specular=0.2, diffuse_scale=0.0)
    metal  = _base_props(base_color=(0.9, 0.9, 0.9), metallic=1.0, specular=0.2, diffuse_scale=0.0)
    _, _, refl_d, *_ = compute_intensity(dielec, cos_i=0.95, R=3.0, cfg=cfg)
    _, _, refl_m, *_ = compute_intensity(metal,  cos_i=0.95, R=3.0, cfg=cfg)
    assert refl_m > refl_d


def test_ior_preferred_over_specular_when_enabled():
    cfg = LidarConfig()
    cfg.prefer_ior = True
    props = _base_props(specular=0.1)
    # low specular but high IOR should yield higher reflectivity when prefer_ior=True
    p1 = dict(props, ior=1.8)
    _, _, refl1, *_ = compute_intensity(p1, cos_i=0.98, R=3.0, cfg=cfg)
    cfg.prefer_ior = False
    _, _, refl2, *_ = compute_intensity(props, cos_i=0.98, R=3.0, cfg=cfg)
    assert refl1 > refl2


def test_energy_conservation_bound():
    cfg = LidarConfig()
    props = _base_props(metallic=0.2, roughness=0.25, specular=0.55, transmission=0.35)
    _, _, refl, Tmat, _ = compute_intensity(props, cos_i=0.9, R=3.0, cfg=cfg)
    assert (refl + Tmat) <= 1.000001


def test_reflectivity_decreases_with_angle():
    cfg = LidarConfig()
    props = _base_props(metallic=0.0, specular=0.3, roughness=0.2, transmission=0.0)
    _, _, refl0, *_ = compute_intensity(props, cos_i=1.0, R=3.0, cfg=cfg)
    _, _, refl1, *_ = compute_intensity(props, cos_i=math.cos(math.radians(60.0)), R=3.0, cfg=cfg)
    assert refl1 < refl0


def test_alpha_coverage_scales_final_intensity_non_clip():
    cfg = LidarConfig()
    props = _base_props()
    # Set explicit opacity (coverage) and ensure it is reported via alpha_cov, then apply to intensity
    props["opacity"] = 0.4
    I0, _, _, _, alpha_cov = compute_intensity(props, cos_i=0.9, R=4.0, cfg=cfg)
    assert abs(alpha_cov - 0.4) < 1e-6
    final_intensity = I0 * alpha_cov
    assert final_intensity < I0 and final_intensity >= 0.0
