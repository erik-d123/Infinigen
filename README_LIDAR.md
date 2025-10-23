# Infinigen LiDAR Ground Truth Generator

This tool generates LiDAR ground truth for Infinigen indoor scenes with indoor-focused defaults (close-range hits enabled, no atmospheric attenuation, dense returns). Run commands from the repository root unless otherwise noted.

## Features

- Indoor-oriented sensor presets: `VLP-16`, `HDL-32E`, `HDL-64E`, `OS1-128`
- Material-aware intensity model (diffuse + specular) with optional percentile auto exposure
- Rolling-shutter and continuous-spin timing with configurable temporal sub-sampling
- Optional override for azimuth column count and output frame (sensor / camera / world)
- Interactive Open3D viewer with ring/intensity coloring and trajectory overlay
- Exports timestamps, TUM poses, metadata JSON, and PLY point clouds with return power, range, normals per frame (ASCII or binary)

## Usage

### Quickstart Script

```bash
./generate_and_view_lidar.sh path/to/scene.blend
```

The script:
- Creates a timestamped output directory `outputs/{preset}/{scene_name}/{YYYYmmdd_HHMMSS}`
- Launches Blender in background mode with the LiDAR generator
- Prints ready-to-run viewer commands

Optional positional arguments (in order):
- `path/to/scene.blend`
- `output_dir` (overrides automatic path)
- `frames` (e.g. `1-48` or `1,5,10`)
- `camera` object name (default `Camera`)
- `preset` (`VLP-16`, `HDL-32E`, `HDL-64E`, `OS1-128`)
- `force_azimuth_steps` (integer override for azimuth columns)

### Direct CLI

```bash
python -m infinigen.launch_blender --background --python lidar/lidar_generator.py -- \
  path/to/scene.blend \
  --output_dir outputs/my_scan \
  --frames 1-48 \
  --camera Camera \
  --preset VLP-16 \
  --ply-frame sensor \
  --force-azimuth-steps 1800 \
  --seed 0
```

Key arguments:
- `--output_dir`: Destination directory (auto-generated if omitted)
- `--frames`: Single value, comma list, or inclusive range
- `--camera`: Camera treated as the LiDAR sensor (first camera by default)
- `--preset`: Sensor preset loaded from `lidar_config.py`
- `--force-azimuth-steps`: Explicit azimuth column count
- `--ply-frame`: Output frame for PLYs (`sensor`, `camera`, `world`)
- `--secondary`: Enable pass-through secondary returns for transmissive surfaces
- `--secondary-extinction`: Beer–Lambert extinction coefficient (1/m) applied when `--secondary` is active
- `--auto-expose`: Enable percentile-based per-frame scaling for the `intensity` column (default is stable, physically based `return_power`)
- `--secondary-min-cos`: Minimum cosine of incidence needed to spawn a pass-through return (default 0.95)
- `--subframes`: Number of temporal pose samples per frame when approximating rolling shutter (default 1)
- `--ply-binary`: Emit binary PLY files instead of ASCII
- `--seed`: Seed for numpy/random (continuous spin still advances phase per frame)

### Viewer

```bash
python lidar/lidar_viewer.py path/to/output_dir --color intensity --view world
```

Parameters:
- `path/to/output_dir`: Directory produced by the generator (must contain `lidar_frame_*.ply`)
- `--color`: Initial coloring mode (`intensity` or `ring`)
- `--view`: Initial viewpoint (`world` or `camera`; `--camera-view` is a shortcut)
- `--frame`: Load a specific frame immediately
- `--no-trajectory`: Hide the camera path overlay

## Intensity & Indoor Defaults

Defaults are tuned for indoor scanning:

1. Distance falloff of `1 / r^2` (physical inverse-square; override as needed)
2. Optional per-frame auto exposure: 95th percentile mapped to intensity 200
3. Principled BSDF sampling for diffuse/specular return power with simple image-texture lookups and transmissive attenuation
4. Minimum range 5 cm to retain close geometry
5. No random dropout; grazing acceptance allows all non-backfacing hits
6. Percentile-based coloring in the viewer for stable contrast

## Outputs

- `lidar_frame_XXXX.ply`: Per-frame point clouds in the chosen frame (with intensity, return power, range, normals)
- `lidar_config.json`: Serialized `LidarConfig` used for the run
- `frame_metadata.json`: Per-frame point counts and intensity scale factors
- `trajectory.json`: Camera translations indexed by frame
- `timestamps.txt`: Seconds from the first frame (derived from scene FPS)
- `poses_tum.txt`: TUM-format poses `timestamp tx ty tz qx qy qz qw`
> The `intensity` column in the PLY is per-frame scaled for visualization. Use the `return_power` float column (and `range_m`) for training or quantitative analysis. The `transmittance` column records per-return energy after glass attenuation (auto exposure only adjusts the 8-bit `intensity`).

### PLY Structure

```
ply
format ascii 1.0
comment Lidar frame 0042
element vertex NNNNN
property float x
property float y
property float z
property uchar intensity
property ushort ring
property float azimuth
property float elevation
property float time_offset
property uchar return_id
property uchar num_returns
property float range_m
property float cos_incidence    # when plyfile installed
property uchar mat_class        # when plyfile installed
property float return_power     # when plyfile installed
property float transmittance    # when pass-through enabled (or legacy exposure scale)
end_header
...
```

Coordinate frames:
- `sensor`: +X forward, +Y left, +Z up (ROS-style; default)
- `camera`: Blender camera space (+X right, +Y up, -Z forward)
- `world`: Blender world coordinates

## Viewer Controls

- Enter frame number: jump to that frame
- `n` / `+`: next frame
- `p` / `-`: previous frame
- `f`: first frame
- `l`: last frame
- `t`: toggle trajectory
- `c`: toggle coloring modes
- `v`: toggle world <-> camera view
- `q`: quit

## References and Documentation

### Velodyne VLP-16 Specifications
- [Velodyne VLP-16 User Manual](https://velodynelidar.com/wp-content/uploads/2019/12/63-9243-Rev-E-VLP-16-User-Manual.pdf)
- [VLP-16 Datasheet](https://velodynelidar.com/products/puck/)
- [Velodyne Sensor Angle Mapping](https://velodynelidar.com/wp-content/uploads/2019/09/PuckChannelMapping.pdf)

### LiDAR Intensity Calculation
- [On Intensity and Range Calibration of Velodyne's Puck Sensor](https://www.mdpi.com/1424-8220/20/11/3217)
- [Intensity Calibration for Automated Segmentation of 3D LiDAR Data](https://ieeexplore.ieee.org/document/8675366)
- [Physics-Based Intensity Correction for Point Cloud Data](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLII-3/413/2018/isprs-archives-XLII-3-413-2018.pdf)

### Ray Casting and Point Cloud Generation
- [Open3D Documentation](http://www.open3d.org/docs/release/)
- [Blender Python API: Ray Casting](https://docs.blender.org/api/current/bpy.types.Scene.html#bpy.types.Scene.ray_cast)
- [PLY File Format](http://paulbourke.net/dataformats/ply/)

### Related Research
- [Sensor-Realistic LiDAR Simulation](https://arxiv.org/pdf/2208.05961.pdf)
- [Simulating LiDAR Point Clouds for Autonomous Driving](https://arxiv.org/pdf/2008.08439.pdf)
- [LiDAR Intensity Calibration: A Review](https://www.mdpi.com/2072-4292/12/17/2697)
