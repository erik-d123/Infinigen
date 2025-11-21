# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.
# Consolidated LiDAR Engine

from __future__ import annotations

import math
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy = None
    Vector = None

from .principled_sampler import PrincipledSampleError, PrincipledSampler

# ==============================================================================
# 1. CONFIGURATION (from lidar_config.py)
# ==============================================================================

# Sensor presets tuned for indoor scenes (rings, max_range in meters)
LIDAR_PRESETS: Dict[str, Dict[str, float]] = {
    "OS0-128": {"rings": 128, "max_range": 50.0},
}


@dataclass
class LidarConfig:
    """Runtime configuration for LiDAR generation and raycasting.

    Fields cover sampling resolution, radiometry and exposure, secondary return
    behavior, coordinate frame for PLY export, and Principled sampling knobs.
    """

    # High-level
    preset: str = "OS0-128"
    force_azimuth_steps: Optional[int] = None  # overrides azimuth columns if set
    ply_frame: str = "sensor"  # {"sensor","camera","world"}
    save_ply: bool = True
    ply_binary: bool = False

    # Radiometry and exposure
    distance_power: float = 2.0  # intensity ∝ 1 / r^p
    auto_expose: bool = False
    global_scale: float = 1.0  # used when auto_expose=False
    target_percentile: float = 95.0  # auto exposure
    target_intensity: float = 200.0  # 8-bit target

    # Ranges and angular dropout
    min_range: float = 0.05  # indoor close hits
    max_range: float = 100.0  # overridden by preset
    grazing_dropout_cos_thresh: float = 0.05  # skip shallow hits
    # Mild angle attenuation of specular lobe to account for footprint effects
    specular_angle_power: float = 0.5  # multiplier: R_spec *= cos_i ** k

    # Secondary return controls (pass-through)
    enable_secondary: bool = False
    secondary_min_residual: float = 0.02  # spawn threshold on residual
    secondary_ray_bias: float = 5e-4  # meters; also aliased as 'hit_offset'
    secondary_min_cos: float = 0.95  # ensure near-normal for pass-through
    secondary_merge_eps: float = 0.0  # meters; merge close returns

    # Sensor layout
    rings: int = 16  # overridden by preset
    azimuth_steps: int = 1800  # default column count

    # Internal/derived store for arbitrary extras
    extras: Dict[str, Any] = field(default_factory=dict)

    # Material sampling
    principled_bake_res: int = 1024  # resolution for per-input socket bakes

    def __post_init__(self):
        if p := LIDAR_PRESETS.get(self.preset):
            self.rings = int(p["rings"])
            self.max_range = float(p["max_range"])

        if self.force_azimuth_steps is not None:
            self.azimuth_steps = int(self.force_azimuth_steps)

        self.grazing_dropout_cos_thresh = max(
            0.0, min(1.0, float(self.grazing_dropout_cos_thresh))
        )
        self.min_range = max(1e-4, float(self.min_range))
        self.max_range = max(self.min_range + 1e-3, float(self.max_range))
        self.rings = max(1, int(self.rings))
        self.azimuth_steps = max(8, int(self.azimuth_steps))

    def hit_offset(self) -> float:
        """Backward‑compatible alias used by the raycaster."""
        return self.secondary_ray_bias

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON‑serializable dict (excluding extras)."""
        d = asdict(self)
        # keep a clean surface in saved JSON
        d.pop("extras", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LidarConfig":
        """Create from a dict, storing unknown keys under `extras`."""
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        extras = {k: v for k, v in d.items() if k not in cls.__dataclass_fields__}
        cfg = cls(**known)
        cfg.extras.update(extras)
        return cfg


# ==============================================================================
# 2. SCENE HELPERS (from lidar_scene.py)
# ==============================================================================


def sensor_to_camera_rotation() -> np.ndarray:
    """
    Return R_cs (camera <- sensor).
    Sensor frame: +X forward, +Y left, +Z up.
    Blender camera: +X right, +Y up, -Z forward.

    Mapping:
      sensor +X (forward) -> camera -Z
      sensor +Y (left)    -> camera -X
      sensor +Z (up)      -> camera +Y
    """
    R = np.array(
        [
            [0.0, -1.0, 0.0],  # camera X  <- { -Y_sensor }
            [0.0, 0.0, 1.0],  # camera Y  <- { +Z_sensor }
            [-1.0, 0.0, 0.0],  # camera Z  <- { -X_sensor }
        ],
        dtype=float,
    )
    # v_cam = R_cs @ v_sensor
    return R


def resolve_camera(scene, camera_name: Optional[str] = None):
    """
    Resolve a camera object:
      1) Named object if provided
      2) scene.camera if set
      3) First object of type 'CAMERA'
    """
    assert bpy is not None, "resolve_camera requires Blender"
    if camera_name:
        obj = scene.objects.get(camera_name)
        if obj and obj.type == "CAMERA":
            return obj
    if getattr(scene, "camera", None):
        return scene.camera
    for obj in scene.objects:
        if getattr(obj, "type", "") == "CAMERA":
            return obj
    raise RuntimeError("No camera found in scene")


def setup_scene(
    scene_path: str, camera_name: Optional[str] = None
) -> Tuple[object, object]:
    """
    Open a .blend and return (scene, camera).
    """
    assert bpy is not None, "setup_scene requires Blender"
    bpy.ops.wm.open_mainfile(filepath=str(scene_path))
    sc = bpy.context.scene
    cam = resolve_camera(sc, camera_name)
    return sc, cam


# ==============================================================================
# 4. IO HELPERS (from lidar_io.py)
# ==============================================================================


def world_to_frame_matrix(camera_obj, frame: str = "sensor") -> np.ndarray:
    """Return 4x4 transform world→{world|camera|sensor} for PLY export.

    Sensor frame is defined as +X forward, +Y left, +Z up. Blender camera uses
    +X right, +Y up, -Z forward.
    """
    if frame == "world":
        return np.eye(4, dtype=float)

    # world -> camera
    R_wc = np.array(camera_obj.matrix_world.to_3x3(), dtype=float)
    t_wc = np.array(camera_obj.matrix_world.translation, dtype=float)
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    Twc_inv = np.eye(4, dtype=float)
    Twc_inv[:3, :3] = R_cw
    Twc_inv[:3, 3] = t_cw

    if frame == "camera":
        return Twc_inv

    # camera <- sensor rotation (R_cs)
    R_cs = np.array(sensor_to_camera_rotation(), dtype=float)
    # world -> sensor = (camera -> sensor) @ (world -> camera)
    R_sc = R_cs.T
    Tcw = Twc_inv
    Tsw = np.eye(4, dtype=float)
    Tsw[:3, :3] = R_sc @ Tcw[:3, :3]
    Tsw[:3, 3] = R_sc @ Tcw[:3, 3]
    return Tsw


# Fixed base order; append optional fields if present.
_BASE_LAYOUT = [
    ("x", "f4"),
    ("y", "f4"),
    ("z", "f4"),
    ("intensity", "u1"),
    ("ring", "u2"),
    ("azimuth", "f4"),
    ("elevation", "f4"),
    ("return_id", "u1"),
    ("num_returns", "u1"),
]
_OPT_FIELDS = [
    ("range_m", "f4"),
    ("cos_incidence", "f4"),
    ("mat_class", "u1"),
    ("reflectivity", "f4"),
    ("transmittance", "f4"),
    # normals written if provided as ("normals", Nx3) or ("nx","ny","nz")
]


def _coerce_col(data: Dict, key: str, dtype: str, N: int) -> Optional[np.ndarray]:
    """Fetch and coerce a 1D column from `data` if present and sized for N."""
    if key not in data:
        return None
    arr = np.asarray(data[key])
    if arr.ndim != 1 or arr.shape[0] != N:
        raise ValueError(f"{key}: expected shape ({N},), got {arr.shape}")
    return arr.astype(dtype, copy=False)


def _coerce_points(pts) -> np.ndarray:
    """Validate and coerce a (N, 3) points array."""
    P = np.asarray(pts)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {P.shape}")
    return P.astype("f4", copy=False)


def _detect_normals(data: Dict, N: int) -> Optional[np.ndarray]:
    """Detect normals as an (N, 3) array, supporting both packed and split forms."""
    if "normals" in data:
        n = np.asarray(data["normals"])
        if n.ndim != 2 or n.shape != (N, 3):
            raise ValueError(f"normals must be (N,3), got {n.shape}")
        return n.astype("f4", copy=False)
    # legacy triplets
    have = all(k in data for k in ("nx", "ny", "nz"))
    if have:
        nx = np.asarray(data["nx"]).astype("f4", copy=False)
        ny = np.asarray(data["ny"]).astype("f4", copy=False)
        nz = np.asarray(data["nz"]).astype("f4", copy=False)
        for a in (nx, ny, nz):
            if a.ndim != 1 or a.shape[0] != N:
                raise ValueError("nx/ny/nz must be (N,)")
        return np.stack([nx, ny, nz], axis=1)
    return None


def _build_header(
    N: int, have: Dict[str, bool], have_normals: bool, binary: bool
) -> str:
    """Build a PLY header string for the present columns and format."""
    fmt = "binary_little_endian 1.0" if binary else "ascii 1.0"
    lines = [
        "ply",
        f"format {fmt}",
        f"element vertex {N}",
    ]
    # base props
    lines += [
        "property float x",
        "property float y",
        "property float z",
        "property uchar intensity",
        "property ushort ring",
        "property float azimuth",
        "property float elevation",
        "property uchar return_id",
        "property uchar num_returns",
    ]
    # optional props in canonical order
    if have.get("range_m"):
        lines.append("property float range_m")
    if have.get("cos_incidence"):
        lines.append("property float cos_incidence")
    if have.get("mat_class"):
        lines.append("property uchar mat_class")
    if have.get("reflectivity"):
        lines.append("property float reflectivity")
    if have.get("transmittance"):
        lines.append("property float transmittance")
    if have_normals:
        lines += ["property float nx", "property float ny", "property float nz"]
    lines.append("end_header")
    return "\n".join(lines) + "\n"


def _stack_record_array(
    data: Dict,
) -> Tuple[np.ndarray, Dict[str, bool], Optional[np.ndarray], list[str]]:
    """Column‑stack core and optional attributes into a dense array for writing."""
    P = _coerce_points(data["points"])
    N = P.shape[0]

    cols = [P[:, 0], P[:, 1], P[:, 2]]
    col_types: list[str] = ["xyz", "xyz", "xyz"]  # keep precision formatting for xyz

    # base
    arr = _coerce_col(data, "intensity", "u1", N)
    intensity = arr if arr is not None else np.zeros(N, "u1")
    arr = _coerce_col(data, "ring", "u2", N)
    ring = arr if arr is not None else np.zeros(N, "u2")
    arr = _coerce_col(data, "azimuth", "f4", N)
    az = arr if arr is not None else np.zeros(N, "f4")
    arr = _coerce_col(data, "elevation", "f4", N)
    el = arr if arr is not None else np.zeros(N, "f4")
    arr = _coerce_col(data, "return_id", "u1", N)
    rid = arr if arr is not None else np.ones(N, "u1")
    arr = _coerce_col(data, "num_returns", "u1", N)
    nret = arr if arr is not None else np.ones(N, "u1")

    cols += [intensity, ring, az, el, rid, nret]
    col_types += ["i", "i", "f", "f", "i", "i"]

    # optionals
    have = {}

    def _is_int_dtype(dt: str) -> bool:
        return dt.lower().startswith(("u", "i"))

    for k, dt in _OPT_FIELDS:
        arr_opt = _coerce_col(data, k, dt, N)
        have[k] = arr_opt is not None
        if arr_opt is not None:
            cols.append(arr_opt)
            col_types.append("i" if _is_int_dtype(dt) else "f")

    normals = _detect_normals(data, N)
    if normals is not None:
        cols += [normals[:, 0], normals[:, 1], normals[:, 2]]
        col_types += ["f", "f", "f"]

    rec = np.column_stack(cols)
    return rec, have, normals, col_types


def save_ply(
    path: str | Path, data: Dict[str, np.ndarray], binary: bool = False
) -> None:
    """Write a PLY with the fields produced by the LiDAR pipeline.

    Required: points (N,3). Optional fields include intensity, ring, azimuth,
    elevation, return_id, num_returns, range_m, cos_incidence, mat_class,
    reflectivity, transmittance, and normals (packed or split).
    """
    path = Path(path)
    if "points" not in data:
        raise ValueError("save_ply: 'points' (N,3) array is required")

    rec, have, normals, col_types = _stack_record_array(data)
    N = rec.shape[0]
    header = _build_header(N, have, normals is not None, binary)

    if not binary:
        # ASCII writer
        with path.open("w", encoding="utf-8") as fh:
            fh.write(header)
            # Write rows with exact number of columns
            for row in rec:
                out = []
                for v, kind in zip(row, col_types):
                    if kind == "xyz":
                        out.append(f"{float(v):.8f}")
                    elif kind == "i":
                        out.append(str(int(v)))
                    else:
                        out.append(str(float(v)))
                fh.write(" ".join(out) + "\n")
        return

    # Binary little-endian
    with path.open("wb") as fh:
        fh.write(header.encode("ascii"))
        # Build per-row struct format based on actual columns present
        fmt = "<"  # little-endian
        # x,y,z
        fmt += "fff"
        # intensity(u1), ring(u2)
        fmt += "BH"
        # azimuth, elevation
        fmt += "ff"
        # return_id, num_returns
        fmt += "BB"
        # optionals in canonical order
        if have.get("range_m"):
            fmt += "f"
        if have.get("cos_incidence"):
            fmt += "f"
        if have.get("mat_class"):
            fmt += "B"
        if have.get("reflectivity"):
            fmt += "f"
        if have.get("transmittance"):
            fmt += "f"
        if normals is not None:
            fmt += "fff"

        pack = struct.Struct(fmt).pack
        # Iterate rows; map types to python scalars
        for row in rec:
            vals = []
            # x,y,z
            vals += [float(row[0]), float(row[1]), float(row[2])]
            # intensity, ring
            vals += [int(row[3]) & 0xFF, int(row[4]) & 0xFFFF]
            # azimuth, elevation
            vals += [float(row[5]), float(row[6])]
            # return_id, num_returns
            vals += [int(row[7]) & 0xFF, int(row[8]) & 0xFF]
            # optionals
            c = 9
            if have.get("range_m"):
                vals.append(float(row[c]))
                c += 1
            if have.get("cos_incidence"):
                vals.append(float(row[c]))
                c += 1
            if have.get("mat_class"):
                vals.append(int(row[c]) & 0xFF)
                c += 1
            if have.get("reflectivity"):
                vals.append(float(row[c]))
                c += 1
            if have.get("transmittance"):
                vals.append(float(row[c]))
                c += 1
            if normals is not None:
                vals += [float(row[c]), float(row[c + 1]), float(row[c + 2])]
                c += 3
            fh.write(pack(*vals))


# ==============================================================================
# 5. INTENSITY MODEL (from intensity_model.py)
# ==============================================================================


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
    """Return per-hit material properties for the intensity model."""
    _require(cfg, "extract_material_properties requires cfg")
    _require(hit_world is not None, "hit_world required for UV sampling")

    try:
        sampled = PrincipledSampler.get().sample(
            obj, depsgraph, poly_index, hit_world, cfg
        )
    except PrincipledSampleError as err:
        # Fallback: Polygon has no material - use default gray diffuse
        print(
            f"WARNING: {getattr(obj, 'name', '<unknown>')}: {err}. Using default gray diffuse."
        )
        sampled = {
            "base_color": (0.5, 0.5, 0.5),
            "metallic": 0.0,
            "roughness": 0.5,
            "transmission": 0.0,
            "transmission_roughness": 0.0,
            "coverage": 1.0,
            "alpha_mode": "OPAQUE",
            "alpha_clip": 0.5,
            "specular": 0.5,
        }

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
        raise ValueError("Principled material missing IOR or Specular value")

    # Pass alpha_linked if present
    if "alpha_linked" in sampled:
        props["alpha_linked"] = sampled["alpha_linked"]

    props["is_glass_hint"] = bool(
        float(props.get("transmission", 0.0)) > 0.5 or float(props["opacity"]) < 0.5
    )
    return props


def compute_intensity(props: Dict, cos_i: float, R: float, cfg):
    """Compact LiDAR radiometry from material properties and incidence.

    Returns a tuple of:
    - intensity: range‑attenuated return power (pre‑alpha)
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
    diffuse_scale = max(0.0, float(props.get("diffuse_scale", 1.0)))
    specular_scale = max(0.0, float(props.get("specular_scale", 1.0)))

    # Coverage must be supplied by sampler (coverage or DIFFUSE alpha)
    alpha_cov = _saturate(props.get("opacity", 1.0))

    # Fresnel base: Principled-style mix between dielectric F0 and metallic-tinted base color
    # Compute dielectric F0 from IOR when available, else from Specular slider, else default ~4%
    if ior and ior > 1.0:
        F0_dielectric = ((ior - 1.0) / (ior + 1.0)) ** 2
    else:
        F0_dielectric = 0.08 * (specular if specular is not None else 0.5)
    F0_dielectric = max(0.0, min(1.0, F0_dielectric))
    F0_rgb = tuple((1.0 - metallic) * F0_dielectric + metallic * c for c in base_rgb)

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
    # Enforce a small minimum albedo (2%) to prevent zero intensity if material sampling fails (returns black)
    diffuse_albedo = max(0.02, min(1.0, props.get("diffuse_albedo", _luma(base_rgb))))
    R_diff = (1.0 - metallic) * (1.0 - F) * diffuse_albedo * cos_i * diffuse_scale
    R_diff = max(0.0, R_diff)

    # Transmission and opaque reflectance
    T_mat = _saturate((1.0 - metallic) * transmission)

    # Compose reflectivity: specular persists even when material is transmissive;
    # attenuate only the diffuse term by transmission.
    reflectivity = _saturate(R_spec + (1.0 - T_mat) * R_diff)

    # Range falloff; alpha will be applied by caller (raycaster)
    # Numerically gentle range falloff; allow epsilon for indoor distances
    eps = float(getattr(cfg, "range_epsilon", 0.0)) if cfg is not None else 0.0
    denom = (
        (distance * distance + max(0.0, eps)) ** (dist_p / 2.0)
        if eps > 0.0
        else (distance**dist_p)
    )
    intensity = reflectivity / max(1e-9, denom)

    return (
        float(intensity),
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


# ==============================================================================
# 6. RAYCASTING (from lidar_raycast.py)
# ==============================================================================


def _unit(v: np.ndarray) -> np.ndarray:
    """Return a unit vector (no-op for zero length)."""
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _clip01(x: np.ndarray) -> np.ndarray:
    """Clamp array values to [0, 1]."""
    return np.clip(x, 0.0, 1.0)


def _percentile_scale(raw_pos: np.ndarray, pct: float, target_u8: float) -> float:
    """Scale such that the pct‑th percentile maps to target_u8/255."""
    p = float(np.percentile(raw_pos, pct))
    if p <= 1e-12:
        return 0.0
    return (target_u8 / 255.0) / p


def _compute_cos_i(normal: np.ndarray, ray_dir: np.ndarray) -> float:
    """Cosine of incidence given a geometric normal and ray direction.

    The ray_dir points from sensor origin into the scene; incidence uses
    the negative ray direction.
    """
    return float(max(0.0, min(1.0, -np.dot(_unit(ray_dir), _unit(normal)))))


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
    # Elevation fan: OS0-128 uses 90 degree vertical FOV (+45 to -45)
    # If other presets were supported, we would switch on cfg.preset here.
    elev = np.linspace(-45.0, 45.0, rings) * (math.pi / 180.0)
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

    # Force view layer update to ensure evaluated objects (CoW) are up-to-date
    # This fixes tests that modify materials between raycasts without explicit updates.
    if bpy.context.view_layer:
        bpy.context.view_layer.update()

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

    # Helper: try a secondary pass‑through and return a dict or None
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
        props2 = extract_material_properties(
            obj2, int(face_index2), depsgraph, loc2, cfg
        )
        I0_2, refl0_2, T2, alpha2 = compute_intensity(props2, cos_i2, r2, cfg)
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
        I0, refl0, T_mat, alpha_cov = compute_intensity(props, cos_i, r, cfg)

        # Alpha handling:
        # - CLIP: cull when coverage below threshold; otherwise do not scale
        # - BLEND/HASHED: never cull by threshold; scale by coverage
        alpha_mode = str(props.get("alpha_mode", "BLEND")).upper()
        alpha_clip = float(props.get("alpha_clip", 0.5))

        if alpha_mode in ("CLIP", "HASHED"):
            if alpha_cov < alpha_clip:
                continue
            alpha_apply = 1.0
        elif alpha_mode == "BLEND":
            # Only apply coverage scaling if the Alpha socket is actually linked.
            alpha_linked = bool(props.get("alpha_linked", False))
            alpha_apply = alpha_cov if alpha_linked else 1.0
        else:
            # OPAQUE or unknown modes: do not scale by coverage
            alpha_apply = 1.0

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
            # Simplified: use transmittance as the primary driver for secondary energy
            residual = float(T_mat * alpha_apply)
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
