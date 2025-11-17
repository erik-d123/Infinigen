"""Configuration and sensor presets for indoor LiDAR generation (baked‑only).

LidarConfig intentionally omits any Principled/material fallbacks. All
material signals must come from exporter outputs (baked textures + sidecars).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

# Sensor presets tuned for indoor scenes (rings, max_range in meters)
LIDAR_PRESETS: Dict[str, Dict[str, float]] = {
    "VLP-16": {"rings": 16, "max_range": 100.0},
    "HDL-32E": {"rings": 32, "max_range": 120.0},
    "HDL-64E": {"rings": 64, "max_range": 120.0},
    "OS1-128": {"rings": 128, "max_range": 120.0},
}


@dataclass
class LidarConfig:
    """Runtime configuration for LiDAR generation and raycasting.

    Fields cover sampling resolution, radiometry and exposure, secondary return
    behavior, coordinate frame for PLY export, and baked material sampling.
    """

    # High-level
    preset: str = "VLP-16"
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

    # Material sampling (baked-only: textures baked by infinigen/tools/export.py)
    export_bake_dir: Optional[str] = None  # path to baked textures folder (required)

    def __post_init__(self):
        p = LIDAR_PRESETS.get(self.preset)
        if p:
            self.rings = int(p["rings"])
            self.max_range = float(p["max_range"])
        if self.force_azimuth_steps is not None:
            self.azimuth_steps = int(self.force_azimuth_steps)
        self.grazing_dropout_cos_thresh = float(
            min(1.0, max(0.0, self.grazing_dropout_cos_thresh))
        )
        self.min_range = float(max(1e-4, self.min_range))
        self.max_range = float(max(self.min_range + 1e-3, self.max_range))
        self.rings = int(max(1, self.rings))
        self.azimuth_steps = int(max(8, self.azimuth_steps))

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
