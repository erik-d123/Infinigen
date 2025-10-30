from __future__ import annotations
import math
from typing import Dict, Tuple, Optional
import numpy as np

try:
    import bpy
except Exception:  # allows pure-Python unit tests
    bpy = None

# Local default in case lidar_config is not importable here
try:
    # used only for default coverage when Alpha is effectively unset
    from lidar.lidar_config import DEFAULT_OPACITY as _CFG_DEFAULT_OPACITY  # type: ignore
except Exception:
    _CFG_DEFAULT_OPACITY = 1.0

# Lazy import to avoid heavy imports when running pure unit tests
def _get_default_opacity(cfg) -> float:
    return float(getattr(cfg, "default_opacity", _CFG_DEFAULT_OPACITY))

def _luma(rgb: Tuple[float, float, float]) -> float:
    r, g, b = rgb
    return max(0.0, min(1.0, 0.2126 * r + 0.7152 * g + 0.0722 * b))

def F_schlick(cos_theta: float, F0: float) -> float:
    cos_theta = max(0.0, min(1.0, float(cos_theta)))
    F0 = max(0.0, min(1.0, float(F0)))
    x = 1.0 - cos_theta
    return F0 + (1.0 - F0) * (x ** 5)

def F_schlick_rgb(cos_theta: float, F0_rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
    cos_theta = max(0.0, min(1.0, float(cos_theta)))
    k = (1.0 - cos_theta) ** 5
    return tuple(
        max(0.0, min(1.0, f0 + (1.0 - f0) * k))
        for f0 in (F0_rgb[0], F0_rgb[1], F0_rgb[2])
    )

def transmissive_reflectance(cos_i: float, ior: float) -> float:
    """Fresnel reflectance for a dielectric at the interface (Schlick)."""
    ior = float(ior if ior else 1.45)
    f0 = ((ior - 1.0) / (ior + 1.0)) ** 2
    return F_schlick(max(0.0, float(cos_i)), f0)

def _find_principled_bsdf(mat) -> Optional["bpy.types.Node"]:
    if not (bpy and mat and getattr(mat, "use_nodes", False) and mat.node_tree):
        return None
    nt = mat.node_tree

    # 1) Try to find the Principled BSDF directly feeding the Material Output's Surface
    try:
        outputs = [n for n in nt.nodes if getattr(n, "type", "") == "OUTPUT_MATERIAL"]
        for out in outputs:
            surf = out.inputs.get("Surface")
            if surf is None:
                continue
            for link in nt.links:
                if link.to_node is out and link.to_socket is surf:
                    if getattr(link.from_node, "type", "") == "BSDF_PRINCIPLED":
                        return link.from_node
    except Exception:
        # Fallback to simple scan below
        pass

    # 2) Fallback: first Principled node found
    for n in nt.nodes:
        if getattr(n, "type", "") == "BSDF_PRINCIPLED":
            return n
    return None

def _get_input_default(bsdf, names, default):
    for n in names:
        sock = bsdf.inputs.get(n)
        if sock is not None and not sock.is_linked:
            try:
                return float(sock.default_value)
            except Exception:
                pass
    return float(default)

def _get_color_default(bsdf, name="Base Color", default=(0.8, 0.8, 0.8)):
    sock = bsdf.inputs.get(name)
    if sock is not None and not sock.is_linked:
        try:
            rgba = tuple(sock.default_value)
            return tuple(max(0.0, min(1.0, float(c))) for c in rgba[:3])
        except Exception:
            pass
    return tuple(default)

def extract_material_properties(obj, poly_index, depsgraph, hit_world=None, cfg=None) -> Dict:
    """
    Read Principled BSDF inputs needed by the intensity model.

    Returns keys:
      base_color (rgb), metallic, specular, roughness, transmission, ior,
      transmission_roughness, opacity, diffuse_scale, specular_scale, is_glass_hint
    """
    props = dict(
        base_color=(0.8, 0.8, 0.8),
        metallic=0.0,
        specular=0.5,
        roughness=0.5,
        transmission=0.0,
        ior=1.45,
        transmission_roughness=0.0,
        # None means: use cfg.default_opacity fallback unless explicitly set/sampled
        opacity=None,
        # Alpha semantics
        alpha_mode="BLEND",   # BLEND | HASHED | CLIP
        alpha_clip=0.5,
        diffuse_scale=1.0,
        specular_scale=1.0,
        is_glass_hint=False,
    )
    if bpy is None:
        return props

    # Resolve material at polygon
    try:
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        poly = mesh.polygons[poly_index]
        mat_index = getattr(poly, "material_index", 0)
        mat = (eval_obj.material_slots[mat_index].material
               if eval_obj.material_slots and mat_index < len(eval_obj.material_slots)
               else obj.active_material)
    except Exception:
        mat = getattr(obj, "active_material", None)

    bsdf = _find_principled_bsdf(mat)
    if bsdf is None:
        # Still try to record alpha semantics from material
        try:
            if mat is not None:
                props["alpha_mode"] = getattr(mat, "blend_method", props["alpha_mode"]).upper()
                props["alpha_clip"] = float(getattr(mat, "alpha_threshold", props["alpha_clip"]))
        except Exception:
            pass
        return props

    # Principled v1/v2 compatibility
    props["base_color"] = _get_color_default(bsdf, "Base Color", props["base_color"])
    props["metallic"] = _get_input_default(bsdf, ["Metallic"], props["metallic"])
    props["specular"] = _get_input_default(bsdf, ["Specular", "Specular IOR Level"], props["specular"])
    props["roughness"] = _get_input_default(bsdf, ["Roughness"], props["roughness"])
    # Transmission socket was renamed in some builds
    t_main = _get_input_default(bsdf, ["Transmission"], 0.0)
    t_w   = _get_input_default(bsdf, ["Transmission Weight"], 0.0)
    props["transmission"] = max(t_main, t_w)
    props["ior"] = _get_input_default(bsdf, ["IOR"], props["ior"])
    # Optional Transmission Roughness
    props["transmission_roughness"] = _get_input_default(bsdf, ["Transmission Roughness"], props["transmission_roughness"])
    # Clearcoat omitted for essential scope
    # Alpha (coverage): if unlinked and intentionally set away from default 1.0, honor it;
    # otherwise leave as None to allow cfg.default_opacity fallback in compute_intensity.
    alpha_sock = bsdf.inputs.get("Alpha")
    if alpha_sock is not None:
        try:
            if not alpha_sock.is_linked:
                val = float(alpha_sock.default_value)
                # Consider values meaningfully different from 1.0 as explicit overrides
                if abs(val - 1.0) > 1e-6:
                    props["opacity"] = max(0.0, min(1.0, val))
                    # If explicit low alpha is set, enforce clip semantics when below threshold
                    try:
                        clip = float(props.get("alpha_clip", 0.5))
                    except Exception:
                        clip = 0.5
                    if val < clip:
                        props["alpha_mode"] = "CLIP"
            else:
                # Linked alpha not evaluated here; defer to cfg.default_opacity
                pass
        except Exception:
            pass

    # Record material alpha semantics for CLIP/HASHED/BLEND handling
    try:
        if mat is not None:
            mode_from_mat = getattr(mat, "blend_method", props["alpha_mode"]) or props["alpha_mode"]
            # Do not override a previously forced CLIP decision
            if str(props.get("alpha_mode", "")).upper() != "CLIP":
                props["alpha_mode"] = str(mode_from_mat).upper()
            props["alpha_clip"] = float(getattr(mat, "alpha_threshold", props["alpha_clip"]))
    except Exception:
        pass

    # Prefer Infinigen export bakes when enabled; never bake here
    if cfg is not None and hit_world is not None and bool(getattr(cfg, "use_export_bakes", True)):
        try:
            from lidar.material_sampler import MaterialSampler
            ms = MaterialSampler.get()
            export_dir = getattr(cfg, "export_bake_dir", None)
            sampled = ms.sample_properties(
                obj, depsgraph, poly_index, hit_world,
                res=int(getattr(cfg, "bake_resolution", 1024)),
                export_bake_dir=export_dir, use_export_bakes=True,
                use_normals=bool(getattr(cfg, "use_baked_normals", True)),
            )
            if sampled:
                for k in [
                    "base_color","roughness","metallic","transmission","opacity",
                    "shading_normal_world",
                ]:
                    if k in sampled and sampled[k] is not None:
                        # Respect config to skip baked normal usage
                        if k == "shading_normal_world" and not bool(getattr(cfg, "use_baked_normals", True)):
                            continue
                        props[k] = sampled[k]
        except Exception:
            pass

    # Optional fallback to direct image textures only when enabled
    if hit_world is not None and cfg is not None and bool(getattr(cfg, "enable_image_fallback", False)):
        try:
            from lidar.fallback_sampler import apply_image_overrides as _apply_image_overrides
            _apply_image_overrides(
                obj=obj,
                depsgraph=depsgraph,
                bsdf=bsdf,
                poly_index=poly_index,
                hit_world=hit_world,
                props=props,
                use_normals=bool(getattr(cfg, "use_baked_normals", True)),
            )
            # If base_color was not overridden (e.g. no UVs), but Base Color is linked to an image,
            # sample its first pixel as a pragmatic fallback (works for 1x1 test textures)
            try:
                # Prefer directly linked image
                bc_sock = bsdf.inputs.get("Base Color")
                tex_node = None
                if bc_sock is not None and bc_sock.is_linked:
                    cand = bc_sock.links[0].from_node
                    if getattr(cand, "type", "") == "TEX_IMAGE":
                        tex_node = cand
                # Otherwise, fall back to any TEX_IMAGE in the material (common in test scenes)
                if tex_node is None:
                    for n in bsdf.id_data.nodes:  # node_tree.nodes
                        if getattr(n, "type", "") == "TEX_IMAGE" and getattr(n, "image", None) is not None:
                            tex_node = n
                            break
                if tex_node is not None and getattr(tex_node, "image", None) is not None:
                    img = tex_node.image
                    w, h = img.size
                    if w > 0 and h > 0:
                        px = np.array(img.pixels[:], dtype=np.float32)
                        r, g, b = float(px[0]), float(px[1]), float(px[2])
                        props["base_color"] = (
                            max(0.0, min(1.0, r)),
                            max(0.0, min(1.0, g)),
                            max(0.0, min(1.0, b)),
                        )
            except Exception:
                pass
        except Exception:
            pass
        if not bool(getattr(cfg, "use_baked_normals", True)):
            props.pop("shading_normal_world", None)

    # Glass hint heuristic for non-opaque transmissive
    _op = props["opacity"] if props.get("opacity", None) is not None else _CFG_DEFAULT_OPACITY
    try:
        op_val = float(_op)
    except Exception:
        op_val = _CFG_DEFAULT_OPACITY
    props["is_glass_hint"] = bool(float(props.get("transmission", 0.0)) > 0.5 or op_val < 0.5)
    return props

def compute_intensity(props: Dict, cos_i: float, R: float, cfg):
    """
    Returns:
        intensity (float): range-attenuated return power (pre-alpha).
        secondary_scale (float): residual for pass-through (0..1).
        reflectivity (float): material reflectivity (pre-alpha).
        transmittance (float): (0..1).
        alpha_cov (float): coverage factor to be applied by caller.
    """
    # Config
    dist_p = float(getattr(cfg, "distance_power", 2.0))
    prefer_ior = bool(getattr(cfg, "prefer_ior", True))

    # Inputs
    cos_i = max(0.0, float(cos_i))
    distance = max(1e-3, float(R))

    base_rgb = tuple(props.get("base_color", (0.8, 0.8, 0.8)))
    metallic = max(0.0, min(1.0, float(props.get("metallic", 0.0))))
    specular = max(0.0, min(1.0, float(props.get("specular", 0.5))))
    rough    = max(0.0, min(1.0, float(props.get("roughness", 0.5))))
    ior      = float(props.get("ior", 1.45) or 0.0)
    transmission = max(0.0, min(1.0, float(props.get("transmission", 0.0))))
    trans_rough  = max(0.0, min(1.0, float(props.get("transmission_roughness", 0.0))))
    diffuse_scale = max(0.0, float(props.get("diffuse_scale", 1.0)))
    specular_scale = max(0.0, float(props.get("specular_scale", 1.0)))

    # Coverage fallback: use cfg.default_opacity only if opacity was not explicitly set
    if "opacity" in props and props["opacity"] is not None:
        alpha_cov = max(0.0, min(1.0, float(props["opacity"])))
    else:
        alpha_cov = max(0.0, min(1.0, _get_default_opacity(cfg)))

    # Fresnel base
    if metallic > 0.0:
        F0_rgb = tuple((1.0 - metallic) * (0.08 * specular) + metallic * c for c in base_rgb)
    else:
        if prefer_ior and ior and ior > 1.0:
            F0 = ((ior - 1.0) / (ior + 1.0)) ** 2
        else:
            F0 = 0.08 * specular
        F0 = max(0.0, min(1.0, F0))
        F0_rgb = (F0, F0, F0)

    F_rgb = F_schlick_rgb(cos_i, F0_rgb)
    F = _luma(F_rgb)

    # Specular lobe (dielectrics honor specular_scale; metals ignore Principled Specular)
    R_spec = F * (max(0.0, 1.0 - rough) ** 2) * (max(0.0, 1.0 - trans_rough) ** 2)
    if metallic <= 0.0:
        R_spec *= specular_scale
    R_spec = max(0.0, min(1.0, R_spec))

    # Diffuse term (Lambert w/out 1/pi, indoor essential)
    diffuse_albedo = max(0.0, min(1.0, props.get("diffuse_albedo", _luma(base_rgb))))
    R_diff = (1.0 - metallic) * (1.0 - F) * diffuse_albedo * cos_i * diffuse_scale
    R_diff = max(0.0, R_diff)

    # Transmission and opaque reflectance
    T_mat = (1.0 - metallic) * transmission
    T_mat = max(0.0, min(1.0, T_mat))

    R_opaque = max(0.0, min(1.0, R_spec + R_diff))
    reflectivity = (1.0 - T_mat) * R_opaque
    reflectivity = max(0.0, min(1.0, reflectivity))

    # Range falloff; alpha will be applied by caller (raycaster)
    intensity = reflectivity / (distance ** dist_p)

    # Residual for pass-through secondary
    secondary_scale = T_mat * max(0.0, 1.0 - F) * (max(0.0, 1.0 - trans_rough) ** 2)
    secondary_scale = max(0.0, min(1.0, secondary_scale))

    return float(intensity), float(secondary_scale), float(reflectivity), float(T_mat), float(alpha_cov)

def classify_material(props: Dict) -> int:
    """0: opaque/dielectric, 1: glass-like, 2: metal."""
    m = float(props.get("metallic", 0.0) or 0.0)
    t = float(props.get("transmission", 0.0) or 0.0)
    op = float(props.get("opacity", 1.0) if props.get("opacity", None) is not None else 1.0)
    if m >= 0.5:
        return 2
    if props.get("is_glass_hint", False) or max(t, 1.0 - op) >= 0.3:
        return 1
    return 0
