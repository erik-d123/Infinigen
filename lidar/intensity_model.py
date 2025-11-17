"""Material sampling and compact reflectance model for indoor LiDAR (baked‑only).

Runtime reads only exporter outputs:
 - Per-hit PBR values from baked textures via MaterialSampler
 - Per-material semantics from sidecar JSONs (alpha_mode/clip, ior/specular, etc.)

We never evaluate Blender node graphs at runtime.

Radiometry:
 - Fresnel via Schlick with metallic mixing and roughness shaping
 - Lambertian diffuse (no 1/pi) for a compact indoor model
 - Transmission term and residual for optional pass-through secondary
 - Alpha semantics are applied once by the raycaster
"""

from __future__ import annotations

from typing import Dict, Tuple


def _require(x, msg: str):
    if x is None:
        raise ValueError(msg)
    return x


def _saturate(x: float) -> float:
    """Clamp a scalar to [0, 1]."""
    return max(0.0, min(1.0, float(x)))


def _luma(rgb: Tuple[float, float, float]) -> float:
    """Compute perceptual luma from RGB in [0,1]."""
    r, g, b = rgb
    return max(0.0, min(1.0, 0.2126 * r + 0.7152 * g + 0.0722 * b))


def F_schlick(cos_theta: float, F0: float) -> float:
    """Schlick Fresnel approximation for a scalar base reflectance F0."""
    cos_theta = max(0.0, min(1.0, float(cos_theta)))
    F0 = max(0.0, min(1.0, float(F0)))
    x = 1.0 - cos_theta
    return F0 + (1.0 - F0) * (x**5)


def F_schlick_rgb(
    cos_theta: float, F0_rgb: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """Schlick Fresnel applied per‑channel for RGB F0."""
    cos_theta = max(0.0, min(1.0, float(cos_theta)))
    k = (1.0 - cos_theta) ** 5
    return tuple(
        max(0.0, min(1.0, f0 + (1.0 - f0) * k))
        for f0 in (F0_rgb[0], F0_rgb[1], F0_rgb[2])
    )


def transmissive_reflectance(cos_i: float, ior: float) -> float:
    """Fresnel reflectance for a dielectric (Schlick form)."""
    ior = float(ior if ior else 1.45)
    f0 = ((ior - 1.0) / (ior + 1.0)) ** 2
    return F_schlick(max(0.0, float(cos_i)), f0)


def extract_material_properties(
    obj, poly_index, depsgraph, hit_world=None, cfg=None
) -> Dict:
    """Return baked-only properties required by the intensity model.

    Requires cfg.export_bake_dir and 'hit_world' to sample UVs.
    """
    from lidar.material_sampler import MaterialSampler

    _require(cfg, "extract_material_properties requires cfg with export_bake_dir")
    _require(
        getattr(cfg, "export_bake_dir", None),
        "export_bake_dir must be set (strict baked-only)",
    )
    _require(hit_world is not None, "hit_world required for UV sampling")

    ms = MaterialSampler.get()
    sampled = ms.sample_properties(
        obj,
        depsgraph,
        poly_index,
        hit_world,
        export_bake_dir=getattr(cfg, "export_bake_dir", None),
    )
    _require(sampled, "MaterialSampler returned no baked properties")

    props: Dict = {
        "base_color": sampled["base_color"],
        "metallic": float(sampled["metallic"]),
        "roughness": float(sampled["roughness"]),
        "transmission": float(sampled.get("transmission", 0.0)),
        "transmission_roughness": float(sampled.get("transmission_roughness", 0.0)),
        "opacity": float(sampled["coverage"]),
        "alpha_mode": str(sampled["alpha_mode"]).upper(),
        "alpha_clip": float(sampled["alpha_clip"]),
        # Scales kept for compatibility in compute_intensity
        "diffuse_scale": 1.0,
        "specular_scale": 1.0,
    }
    if "ior" in sampled:
        props["ior"] = float(sampled["ior"])
    elif "specular" in sampled:
        props["specular"] = float(sampled["specular"])
    else:
        raise ValueError("Sidecar missing ior/specular (strict baked-only)")

    props["is_glass_hint"] = bool(
        float(props.get("transmission", 0.0)) > 0.5 or float(props["opacity"]) < 0.5
    )
    return props


def compute_intensity(props: Dict, cos_i: float, R: float, cfg):
    """Compact LiDAR radiometry from material properties and incidence.

    Returns a tuple of:
    - intensity: range‑attenuated return power (pre‑alpha)
    - secondary_scale: residual energy for pass‑through (0..1)
    - reflectivity: material reflectivity (pre‑alpha)
    - transmittance: material transmission in [0,1]
    - alpha_cov: coverage factor; caller applies CLIP/BLEND semantics
    """
    # Config
    dist_p = float(getattr(cfg, "distance_power", 2.0))

    # Inputs
    cos_i = max(0.0, float(cos_i))
    distance = max(1e-3, float(R))

    base_rgb = tuple(props.get("base_color", (0.8, 0.8, 0.8)))
    metallic = _saturate(props.get("metallic", 0.0))
    specular = props.get("specular", None)
    if specular is not None:
        specular = _saturate(specular)
    rough = _saturate(props.get("roughness", 0.5))
    ior = float(props.get("ior", 1.45) or 0.0)
    transmission = _saturate(props.get("transmission", 0.0))
    trans_rough = _saturate(props.get("transmission_roughness", 0.0))
    diffuse_scale = max(0.0, float(props.get("diffuse_scale", 1.0)))
    specular_scale = max(0.0, float(props.get("specular_scale", 1.0)))

    # Coverage must be supplied by sampler (coverage or DIFFUSE alpha)
    alpha_cov = _saturate(props.get("opacity", 1.0))

    # Fresnel base
    if metallic > 0.0:
        F0_rgb = tuple(
            (1.0 - metallic) * (0.08 * specular) + metallic * c for c in base_rgb
        )
    else:
        if ior and ior > 1.0:
            F0 = ((ior - 1.0) / (ior + 1.0)) ** 2
        else:
            if specular is None:
                raise ValueError("Missing both ior and specular for Fresnel base")
            F0 = 0.08 * specular
        F0 = max(0.0, min(1.0, F0))
        F0_rgb = (F0, F0, F0)

    F_rgb = F_schlick_rgb(cos_i, F0_rgb)
    F = _luma(F_rgb)

    # Specular lobe (dielectrics honor specular_scale; metals ignore Principled Specular)
    # Specular amplitude shaped by roughness, with mild angle attenuation to account for footprint effects
    R_spec = F * (max(0.0, 1.0 - rough) ** 2)
    try:
        k = float(getattr(cfg, "specular_angle_power", 0.5))
    except Exception:
        k = 0.5
    if k > 0.0:
        R_spec *= max(0.0, cos_i) ** k
    if metallic <= 0.0:
        R_spec *= specular_scale
    R_spec = max(0.0, min(1.0, R_spec))

    # Diffuse term (Lambert w/out 1/pi, indoor essential)
    diffuse_albedo = max(0.0, min(1.0, props.get("diffuse_albedo", _luma(base_rgb))))
    R_diff = (1.0 - metallic) * (1.0 - F) * diffuse_albedo * cos_i * diffuse_scale
    R_diff = max(0.0, R_diff)

    # Transmission and opaque reflectance
    T_mat = _saturate((1.0 - metallic) * transmission)

    R_opaque = _saturate(R_spec + R_diff)
    reflectivity = _saturate((1.0 - T_mat) * R_opaque)

    # Range falloff; alpha will be applied by caller (raycaster)
    intensity = reflectivity / (distance**dist_p)

    # Residual for pass-through secondary
    secondary_scale = _saturate(
        T_mat * max(0.0, 1.0 - F) * (max(0.0, 1.0 - trans_rough) ** 2)
    )

    return (
        float(intensity),
        float(secondary_scale),
        float(reflectivity),
        float(T_mat),
        float(alpha_cov),
    )


def classify_material(props: Dict) -> int:
    """Return a coarse material class: 0=dielectric, 1=glass‑like, 2=metal."""
    m = float(props.get("metallic", 0.0) or 0.0)
    t = float(props.get("transmission", 0.0) or 0.0)
    op = float(
        props.get("opacity", 1.0) if props.get("opacity", None) is not None else 1.0
    )
    if m >= 0.5:
        return 2
    if props.get("is_glass_hint", False) or max(t, 1.0 - op) >= 0.3:
        return 1
    return 0
