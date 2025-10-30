from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional

# Alpha fallback when Principled Alpha is unset
DEFAULT_OPACITY: float = 1.0

# Sensor presets tuned for indoor scenes (rings, max_range in meters)
LIDAR_PRESETS: Dict[str, Dict[str, float]] = {
    "VLP-16":   {"rings": 16,  "max_range": 100.0},
    "HDL-32E":  {"rings": 32,  "max_range": 120.0},
    "HDL-64E":  {"rings": 64,  "max_range": 120.0},
    "OS1-128":  {"rings": 128, "max_range": 120.0},
}

@dataclass
class LidarConfig:
    # High-level
    preset: str = "VLP-16"
    force_azimuth_steps: Optional[int] = None           # overrides azimuth columns if set
    ply_frame: str = "sensor"                           # {"sensor","camera","world"}
    save_ply: bool = True
    ply_binary: bool = False

    # Radiometry and exposure
    distance_power: float = 2.0                         # intensity ∝ 1 / r^p
    auto_expose: bool = False
    global_scale: float = 1.0                           # used when auto_expose=False
    target_percentile: float = 95.0                     # auto exposure
    target_intensity: float = 200.0                     # 8-bit target
    default_opacity: float = DEFAULT_OPACITY            # alpha-as-coverage fallback
    prefer_ior: bool = True                             # derive F0 from IOR when present

    # Ranges and angular dropout
    min_range: float = 0.05                             # indoor close hits
    max_range: float = 100.0                            # overridden by preset
    grazing_dropout_cos_thresh: float = 0.05            # skip shallow hits

    # Secondary return controls (pass-through)
    enable_secondary: bool = False
    secondary_min_residual: float = 0.02                # spawn threshold on residual
    secondary_ray_bias: float = 5e-4                    # meters; also aliased as 'hit_offset'
    secondary_min_cos: float = 0.95                     # ensure near-normal for pass-through
    secondary_merge_eps: float = 0.0                    # meters; merge close returns

    # Sensor layout
    rings: int = 16                                     # overridden by preset
    azimuth_steps: int = 1800                           # default column count

    # Internal/derived store for arbitrary extras
    extras: Dict[str, Any] = field(default_factory=dict)

    # Material sampling strategy (prefer Infinigen export bakes; never rebake here)
    use_export_bakes: bool = True                       # use textures baked by infinigen/tools/export.py
    export_bake_dir: Optional[str] = None               # path to baked textures folder (auto-detected or provided)
    # Control whether to use baked/tangent-space normal maps for shading incidence
    use_baked_normals: bool = True
    # Optional: allow direct-texture fallback sampling from Principled image links
    enable_image_fallback: bool = False
    # Legacy flags (no-op): retained for compatibility; LiDAR no longer triggers baking itself
    bake_pbr: bool = False                              # deprecated (LiDAR does not bake)
    bake_resolution: int = 1024                         # deprecated
    bake_normals: bool = True                           # deprecated

    def __post_init__(self):
        p = LIDAR_PRESETS.get(self.preset)
        if p:
            self.rings = int(p["rings"])
            self.max_range = float(p["max_range"])
        if self.force_azimuth_steps is not None:
            self.azimuth_steps = int(self.force_azimuth_steps)
        self.default_opacity = float(min(1.0, max(0.0, self.default_opacity)))
        self.grazing_dropout_cos_thresh = float(min(1.0, max(0.0, self.grazing_dropout_cos_thresh)))
        self.min_range = float(max(1e-4, self.min_range))
        self.max_range = float(max(self.min_range + 1e-3, self.max_range))
        self.rings = int(max(1, self.rings))
        self.azimuth_steps = int(max(8, self.azimuth_steps))

    def hit_offset(self) -> float:
        # Backward-compatible alias used by raycaster
        return self.secondary_ray_bias

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # keep a clean surface in saved JSON
        d.pop("extras", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LidarConfig":
        # Allow unknown keys via extras
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        extras = {k: v for k, v in d.items() if k not in cls.__dataclass_fields__}
        cfg = cls(**known)
        cfg.extras.update(extras)
        return cfg
