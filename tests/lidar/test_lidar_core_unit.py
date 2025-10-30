#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np

from lidar.lidar_config import LidarConfig
from lidar.intensity_model import compute_intensity


def test_default_opacity_fallback_compute_intensity():
    cfg = LidarConfig()
    cfg.default_opacity = 0.25
    props = {
        # leave opacity absent to force fallback
        "base_color": (0.8, 0.8, 0.8),
        "metallic": 0.0,
        "specular": 0.5,
        "roughness": 0.5,
        "transmission": 0.0,
        "transmission_roughness": 0.0,
    }
    _, _, _, _, alpha_cov = compute_intensity(props, cos_i=1.0, R=3.0, cfg=cfg)
    assert abs(alpha_cov - 0.25) < 1e-6


def test_angle_reduces_intensity():
    cfg = LidarConfig()
    cfg.auto_expose = False
    props = {
        "base_color": (0.6, 0.6, 0.6),
        "metallic": 0.0,
        "specular": 0.5,
        "roughness": 0.2,
        "transmission": 0.0,
        "transmission_roughness": 0.0,
    }
    I_n, _, _, _, _ = compute_intensity(props, cos_i=1.0, R=5.0, cfg=cfg)
    I_t, _, _, _, _ = compute_intensity(props, cos_i=math.cos(math.radians(35.0)), R=5.0, cfg=cfg)
    assert I_t < I_n


def test_transmission_reduces_reflectivity_and_secondary_positive():
    cfg = LidarConfig()
    props_opaque = {
        "base_color": (0.8, 0.8, 0.8),
        "metallic": 0.0,
        "specular": 0.5,
        "roughness": 0.1,
        "transmission": 0.0,
        "transmission_roughness": 0.0,
    }
    props_glass = dict(props_opaque)
    props_glass.update({"transmission": 0.9, "transmission_roughness": 0.0})

    _, res_opaque, refl_opaque, T_opaque, _ = compute_intensity(props_opaque, cos_i=0.95, R=3.0, cfg=cfg)
    _, res_glass, refl_glass, T_glass, _ = compute_intensity(props_glass, cos_i=0.95, R=3.0, cfg=cfg)

    assert refl_glass < refl_opaque
    assert T_glass > T_opaque
    assert res_glass > 0.0


# Clearcoat removed from essential scope; no test needed.


def test_energy_conservation_bound():
    cfg = LidarConfig()
    props = {
        "base_color": (0.5, 0.4, 0.3),
        "metallic": 0.2,
        "roughness": 0.25,
        "specular": 0.55,
        "transmission": 0.35,
        "transmission_roughness": 0.0,
    }
    _, _, refl, Tmat, _ = compute_intensity(props, cos_i=0.9, R=3.0, cfg=cfg)
    assert refl + Tmat <= 1.000001
