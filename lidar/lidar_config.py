#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# LiDAR sensor configuration and presets module
# Contains sensor specifications and configuration management

import numpy as np

LIDAR_PRESETS = {
    "VLP-16": {
        "num_elevation": 16,
        "elevation_angles_deg": [
            -15.0,  1.0,  -13.0,  3.0,  -11.0,  5.0,  -9.0,   7.0,
             -7.0,  9.0,   -5.0, 11.0,  -3.0, 13.0,  -1.0,  15.0
        ],
        "min_range": 0.9,         # Manufacturer spec: 0.9m
        "max_range": 100.0,
        "range_accuracy": 0.03,    # ±3cm
        
        "default_rpm": 600.0,      # 10 Hz
        "policy": "velodyne",
        "pps": 300000,            # Points per second
    },
    "HDL-32E": {
        "num_elevation": 32,
        "elevation_angles_deg": [
            -30.67, -9.33, -29.33, -8.00, -28.00, -6.67, -26.67, -5.33,
            -25.33, -4.00, -24.00, -2.67, -22.67, -1.33, -21.33,  0.00,
            -20.00,  1.33, -18.67,  2.67, -17.33,  4.00, -16.00,  5.33,
            -14.67,  6.67, -13.33,  8.00, -12.00,  9.33, -10.67, 10.67
        ],
        "min_range": 1.0,         # Manufacturer spec: 1.0m
        "max_range": 100.0,
        "range_accuracy": 0.02,    # ±2cm
        
        "default_rpm": 600.0,      # 10 Hz
        "policy": "velodyne",
        "pps": 700000,            # Points per second
    },
    "HDL-64E": {
        "num_elevation": 64,
        "elevation_angles_deg": [float(x) for x in np.linspace(2.0, -24.9, 64)],
        "min_range": 0.9,         # Manufacturer spec: 0.9m
        "max_range": 120.0,
        "range_accuracy": 0.02,    # ±2cm
        
        "default_rpm": 600.0,      # 10 Hz
        "policy": "velodyne",
        "pps": 1300000,           # Points per second
    },
    "OS1-128": {
        "num_elevation": 128,
        "vfov_deg": 42.4,         # ±21.2°
        "min_range": 0.5,         # Manufacturer spec: 0.5m
        "max_range": 120.0,
        "range_accuracy": 0.02,    # ±2cm
        
        "default_rpm": 1200.0,     # 20 Hz
        "policy": "ouster",
        "columns_per_rev": 2048,   # Fixed columns per revolution
    }
}

class LidarConfig:
    def __init__(self,
                 preset: str = "VLP-16",
                 force_azimuth_steps: int = None,
                 save_ply: bool = True,
                 auto_expose: bool = False,
                 global_scale: float = 1.0,
                 rpm: float = None,
                 continuous_spin: bool = True,
                 rolling_shutter: bool = True,
                 subframes: int = 1,
                 ply_frame: str = "sensor",  # {camera,sensor,world}
                 enable_secondary: bool = False,
                 secondary_min_residual: float = 0.05,
                 secondary_ray_bias: float = 5e-4,
                 secondary_extinction: float = 0.0,
                 secondary_min_cos: float = 0.95,
                 ply_binary: bool = False,
                 ):

        if preset not in LIDAR_PRESETS:
            raise ValueError(f"Unknown LiDAR preset: {preset}. Available: {list(LIDAR_PRESETS.keys())}")

        preset_data = LIDAR_PRESETS[preset]
        
        # Geometry configuration from preset
        self.preset = preset
        self.num_elevation = preset_data["num_elevation"]
        
        # Handle OS1-128 special case - generate elevation angles from vfov
        if preset == "OS1-128":
            vfov_deg = preset_data["vfov_deg"]
            half_vfov = vfov_deg / 2.0
            self.elevation_angles_deg = list(np.linspace(half_vfov, -half_vfov, self.num_elevation))
        else:
            self.elevation_angles_deg = preset_data["elevation_angles_deg"]
        
        # Determine azimuth steps based on policy
        policy = preset_data["policy"]
        self.rpm = rpm or preset_data["default_rpm"]
        
        if force_azimuth_steps is not None:
            # Override policy with explicit azimuth steps if provided
            self.num_azimuth = force_azimuth_steps
        elif policy == "velodyne":
            # Velodyne: calculate from points-per-second, channel count, and rotation rate
            pps = preset_data["pps"]
            rps = self.rpm / 60.0  # Revolutions per second
            self.num_azimuth = int(round(pps / (self.num_elevation * rps)))
        elif policy == "ouster":
            # Ouster: fixed columns per revolution, independent of RPM
            self.num_azimuth = preset_data["columns_per_rev"]
        else:
            raise ValueError(f"Unknown LiDAR policy: {policy}")

        # Range limits: indoor-only usage → allow very close hits
        self.min_range = 0.05  # meters; low but avoids self-intersections
        self.max_range = preset_data["max_range"]

        # Intensity model parameters (indoor-friendly defaults)
        self.distance_power = 2.0
        self.target_percentile = 95
        self.target_intensity = 200
        self.auto_expose = auto_expose
        self.global_scale = global_scale
        # Atmospheric attenuation disabled for indoor usage
        self.beta_atm = 0.0

        # Sensor timing and motion
        self.rpm = rpm or preset_data["default_rpm"]
        self.continuous_spin = continuous_spin
        self.rolling_shutter = rolling_shutter
        self.subframes = max(1, int(subframes))

        # Output options (kept minimal)
        self.save_ply = save_ply
        self.ply_frame = ply_frame

        # Noise and realism parameters (based on preset accuracy)
        self.range_noise_a = preset_data["range_accuracy"]  # Base range noise from spec
        self.range_noise_b = 0.001         # Range-proportional noise  
        self.intensity_jitter_std = 0.02   # Multiplicative intensity jitter
        # Return as much as possible for indoor scans
        self.dropout_prob = 0.0            # No random dropout by default
        self.grazing_dropout_cos_thresh = 0.05
        # Secondary/pass-through rays
        self.enable_secondary = enable_secondary
        self.secondary_min_residual = secondary_min_residual
        self.secondary_ray_bias = secondary_ray_bias
        self.secondary_extinction = secondary_extinction
        self.secondary_min_cos = secondary_min_cos
        # Backward compatibility alias
        self.rolling_subframes = self.subframes
        self.ply_binary = ply_binary

    def to_dict(self):
        # Convert configuration to serializable dict
        return {
            "preset": self.preset,
            "num_elevation": self.num_elevation,
            "num_azimuth": self.num_azimuth,
            "elevation_angles_deg": self.elevation_angles_deg,
            "min_range": self.min_range,
            "max_range": self.max_range,
            "distance_power": self.distance_power,
            "target_percentile": self.target_percentile,
            "target_intensity": self.target_intensity,
            "auto_expose": self.auto_expose,
            "global_scale": self.global_scale,
            "beta_atm": self.beta_atm,
            "rpm": self.rpm,
            "continuous_spin": self.continuous_spin,
            "rolling_shutter": self.rolling_shutter,
            "subframes": self.subframes,
            "save_ply": self.save_ply,
            "ply_frame": self.ply_frame,
            "range_noise_a": self.range_noise_a,
            "range_noise_b": self.range_noise_b,
            "intensity_jitter_std": self.intensity_jitter_std,
            "dropout_prob": self.dropout_prob,
            "grazing_dropout_cos_thresh": self.grazing_dropout_cos_thresh,
            "enable_secondary": self.enable_secondary,
            "secondary_min_residual": self.secondary_min_residual,
            "secondary_ray_bias": self.secondary_ray_bias,
            "secondary_extinction": self.secondary_extinction,
            "secondary_min_cos": self.secondary_min_cos,
            "rolling_subframes": self.subframes,
            "ply_binary": self.ply_binary,
        }

    def __str__(self):
        return f"{self.preset} indoor: {self.num_elevation}x{self.num_azimuth}, rpm={self.rpm}, rolling={self.rolling_shutter}, spin={self.continuous_spin}"
