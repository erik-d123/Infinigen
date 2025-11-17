import math

from lidar.intensity_model import compute_intensity
from lidar.lidar_config import LidarConfig


def _props(**over):
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


def test_coverage_scales_intensity():
    cfg = LidarConfig()
    base = _props()
    base["opacity"] = 1.0
    I_full, _, _, _, a_full = compute_intensity(base, cos_i=1.0, R=3.0, cfg=cfg)
    base["opacity"] = 0.25
    I_quarter, _, _, _, a_quarter = compute_intensity(base, cos_i=1.0, R=3.0, cfg=cfg)
    # Reflectivity/intensity from compute_intensity are pre-alpha; simulate raycaster scaling
    assert (I_quarter * a_quarter) < (I_full * a_full) and a_quarter < a_full


def test_angle_decreases_intensity():
    cfg = LidarConfig()
    I0, *_ = compute_intensity(_props(), cos_i=1.0, R=5.0, cfg=cfg)
    I1, *_ = compute_intensity(
        _props(), cos_i=math.cos(math.radians(40.0)), R=5.0, cfg=cfg
    )
    assert I1 < I0


def test_distance_falloff():
    cfg = LidarConfig()
    cfg.distance_power = 2.0
    I2, *_ = compute_intensity(_props(), cos_i=1.0, R=2.0, cfg=cfg)
    I4, *_ = compute_intensity(_props(), cos_i=1.0, R=4.0, cfg=cfg)
    assert I2 > 0 and I4 > 0 and (I2 / I4) > 3.5


def test_transmission_reduces_reflectivity_and_residual_positive():
    cfg = LidarConfig()
    _, res0, refl0, T0, _ = compute_intensity(
        _props(transmission=0.0), cos_i=0.95, R=3.0, cfg=cfg
    )
    _, res1, refl1, T1, _ = compute_intensity(
        _props(transmission=0.85), cos_i=0.95, R=3.0, cfg=cfg
    )
    assert refl1 < refl0 and T1 > T0 and res1 > 0.0


def test_metallic_mixing_increases_fresnel():
    cfg = LidarConfig()
    dielec = _props(
        base_color=(0.9, 0.9, 0.9), metallic=0.0, specular=0.2, diffuse_scale=0.0
    )
    metal = _props(
        base_color=(0.9, 0.9, 0.9), metallic=1.0, specular=0.2, diffuse_scale=0.0
    )
    _, _, refl_d, *_ = compute_intensity(dielec, cos_i=0.95, R=3.0, cfg=cfg)
    _, _, refl_m, *_ = compute_intensity(metal, cos_i=0.95, R=3.0, cfg=cfg)
    assert refl_m > refl_d


def test_ior_produces_higher_f0_than_low_specular():
    cfg = LidarConfig()
    props = _props(specular=0.1)
    p1 = dict(props, ior=1.8)
    _, _, refl1, *_ = compute_intensity(p1, cos_i=0.98, R=3.0, cfg=cfg)
    _, _, refl2, *_ = compute_intensity(props, cos_i=0.98, R=3.0, cfg=cfg)
    assert refl1 > refl2


def test_energy_conservation_bound():
    cfg = LidarConfig()
    _, _, refl, Tmat, _ = compute_intensity(
        _props(metallic=0.2, roughness=0.25, specular=0.55, transmission=0.35),
        cos_i=0.9,
        R=3.0,
        cfg=cfg,
    )
    assert (refl + Tmat) <= 1.000001


def test_alpha_coverage_scales_reported():
    cfg = LidarConfig()
    props = _props()
    props["opacity"] = 0.4
    I0, _, _, _, alpha_cov = compute_intensity(props, cos_i=0.9, R=4.0, cfg=cfg)
    assert abs(alpha_cov - 0.4) < 1e-6 and I0 > 0.0
