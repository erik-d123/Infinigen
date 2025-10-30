# Infinigen LiDAR Ground Truth Generator

This tool generates LiDAR ground truth for Infinigen indoor scenes with indoor-focused defaults (close-range hits enabled, no atmospheric attenuation, dense returns). Run commands from the repository root unless otherwise noted.

## Features

- Uses Infinigen export PBR bakes by default (UV + texture maps from procedural materials): Albedo/Base Color, Roughness, Metallic, Transmission, Normal.
- Energy‑preserving reflectivity model (Lambert diffuse + Schlick specular) with metallic mixing and roughness shaping.
- Transmission reduces opaque reflectance and enables a single pass‑through secondary; alpha is coverage (CLIP culls below threshold; BLEND/HASHED uses coverage).
- Indoor‑oriented sensor presets: `VLP-16`, `HDL-32E`, `HDL-64E`, `OS1-128`.
- Percentile auto‑exposure for 8‑bit intensity (float reflectivity retained for training).
- Output frame control (sensor/camera/world); Open3D viewer with ring/intensity/reflectivity coloring.
- Exports timestamps, TUM poses, metadata JSON, and PLY point clouds with reflectivity, transmittance, range, normals per frame (ASCII or binary).

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
python -m infinigen.launch_blender -m lidar.lidar_generator -- \
  path/to/scene.blend \
  --output_dir outputs/my_scan \
  --frames 1-48 \
  --camera Camera \
  --preset VLP-16 \
  --ply-frame sensor \
  --force-azimuth-steps 1800 \
  --no-bake-normals \
  --seed 0
```

### Programmatic Use (inside Blender)

```python
from lidar.lidar_generator import generate_for_scene

generate_for_scene(
    scene_path="outputs/indoors/example/scene.blend",
    output_dir="outputs/lidar/example",
    frames=[1, 2, 3],
    camera_name="Camera",
    cfg_kwargs={"preset": "HDL-32E", "auto_expose": True},
)
```

Key arguments:
- Output and camera
  - `--output_dir`: Destination directory (auto‑generated if omitted)
  - `--frames`: Single value, comma list, or inclusive range
  - `--camera`: Camera treated as the LiDAR sensor (first camera by default)
  - `--ply-frame`: Output frame for PLYs (`sensor`, `camera`, `world`)
- Sensor and resolution
  - `--preset`: Sensor preset loaded from `lidar_config.py`
  - `--force-azimuth-steps`: Explicit azimuth column count
- Export PBR usage (streamlined)
  - `--no-bake-pbr`: Disable use of baked PBR maps entirely.
  - `--export-bake-dir`: Folder with baked textures to sample (defaults to detected scene textures when present)
  - `--no-bake-normals`: Keep baked colors/scalars but skip normal map usage (good quality/speed tradeoff)
  - `enable_image_fallback` (config): Off by default; when enabled, samples direct Principled Image textures to fill gaps (slower, only if you truly need it).
- Radiometry
  - `--secondary`: Enable pass‑through secondary returns for transmissive surfaces
  - `--secondary-min-cos`: Minimum cosine of incidence needed to spawn a pass‑through return (default 0.95)
  - `--auto-expose`: Enable percentile‑based per‑frame scaling for the `intensity` column
- Misc
  - `--ply-binary`: Emit binary PLY files instead of ASCII
  - `--seed`: Seed for numpy/random (continuous spin still advances phase per frame)

### Viewer

```bash
python lidar/lidar_viewer.py path/to/output_dir --color intensity --view world
```

Parameters:
- `path/to/output_dir`: Directory produced by the generator (must contain `lidar_frame_*.ply`)
- `--color`: Initial coloring mode (`intensity`, `reflectivity`, or `ring`)
- `--view`: Initial viewpoint (`world` or `camera`; `--camera-view` is a shortcut)
- `--frame`: Load a specific frame immediately
- `--no-trajectory`: Hide the camera path overlay

## Material Realization & Intensity

By default, LiDAR uses the same PBR maps that Infinigen generates for export (UV + texture baking of procedural materials):

- PBR inputs: Albedo/Base Color (RGB), Roughness (R), Metallic (R), Transmission (R), Normal (tangent‑space RGB). Alpha follows Blender semantics (CLIP threshold vs BLEND/HASHED coverage).
- Shading normal: The baked tangent normal is converted to world space with TBN and used for cos(incidence); backfaces are flipped for correct incidence.
- Reflectivity: Lambert diffuse + Schlick specular with metallic mixing and roughness shaping; optional small clearcoat lobe. Transmission reduces opaque reflectance and supplies residual for a single pass‑through secondary.
- Intensity: `intensity` is 8‑bit (optional percentile auto‑exposure); `reflectivity` is a float channel for training.

### Two Modes

- Export‑bake mode (recommended): bake once per scene (or reuse any prior export) and point `--export-bake-dir` at the textures folder. Fast and consistent: runtime LiDAR only samples textures at UVs.
- Lightweight/no‑bake mode (fastest iteration):
  - `--no-bake-pbr`: disables use of exporter bakes. Optionally enable `enable_image_fallback` in config to sample direct Image textures.
  - `--no-bake-normals`: keep baked colors/scalars but skip tangent normal (use geometric normal incidence).


## Outputs

- `lidar_frame_XXXX.ply`: Per-frame point clouds in the chosen frame (with intensity, reflectivity, range, normals)
- `lidar_config.json`: Serialized `LidarConfig` used for the run
- `frame_metadata.json`: Per-frame point counts and intensity scale factors
- `trajectory.json`: Camera translations indexed by frame
- `timestamps.txt`: Seconds from the first frame (derived from scene FPS)
- `poses_tum.txt`: TUM-format poses `timestamp tx ty tz qx qy qz qw`
> The `intensity` column in the PLY is per-frame scaled for visualization. Use the `reflectivity` float column (and `range_m`) for training or quantitative analysis. The `transmittance` column mirrors the Principled `Transmission` value for the surface hit (auto exposure only adjusts the 8-bit `intensity`).

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
property float reflectivity     # when plyfile installed
property float transmittance    # when pass-through enabled (or legacy exposure scale)
end_header
...
```

Coordinate frames:
- `sensor`: +X forward, +Y left, +Z up (ROS-style; default)
- `camera`: Blender camera space (+X right, +Y up, -Z forward)
- `world`: Blender world coordinates

## Using Infinigen Export Bakes

Infinigen’s exporter (infinigen/tools/export.py) bakes procedural materials to PBR maps for portability. LiDAR can reuse the same maps:

1) Pre‑bake once per scene (fastest LiDAR runtime):

```bash
python -m infinigen.tools.export \
  --input_folder outputs/MYJOB/SEED/coarse \
  --output_folder outputs/MYJOB/SEED/export \
  -f usdc -r 1024
```

This produces `outputs/MYJOB/SEED/export/textures/{object}_{BAKE}.png`. Then run LiDAR with:

```bash
python -m infinigen.launch_blender --background --python lidar/lidar_generator.py -- \
  outputs/MYJOB/SEED/coarse/scene.blend \
  --export-bake-dir outputs/MYJOB/SEED/export/textures \
  --frames 1-16 --camera Camera --preset VLP-16
```

2) Or, run without pre‑bake:
- `--no-bake-pbr`: sample direct textures/defaults (fastest, less accurate on complex materials).
- `--no-bake-normals`: skip normal maps (good quality/speed tradeoff).

Expected bake names: `{object_clean_name}_{DIFFUSE|ROUGHNESS|NORMAL|METAL|TRANSMISSION}.png`.

Note on names: the exporter cleans object/material names by replacing spaces and dots with underscores. LiDAR’s sampler applies the same cleaning when looking up `{object_clean_name}_*.png`. If you rename objects after export, the sampler may miss maps. Re-export or pass a consistent name mapping.

Optional geometry alignment: for small occlusion differences on displaced indoor walls, enable room ocmeshing during scene generation so LiDAR and render raycast the same realized geometry. Use an override when generating indoors scenes:

```
python -m infinigen.datagen.manage_jobs ... \
  --pipeline_overrides get_cmd.driver_script='infinigen_examples.generate_indoors' \
  --overrides compose_indoors.enable_ocmesh_room=True
```

## Pipeline Integration (Preview)

We plan to add a `lidar.gin` and `queue_lidar` so LiDAR runs as a camera‑dependent task in manage_jobs. A recommended setup is:
- Global task: run the exporter bake once per scene (`bake_scene`) to a textures folder.
- Camera task: run LiDAR with `--export-bake-dir` pointing to that folder.

This makes LiDAR outputs slot alongside images/GT with consistent material realization and performance.

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

## Tests

Unit tests and Blender integration tests live in `tests/lidar/`:
- Unit: default opacity fallback, angle vs intensity, transmission vs reflectivity and secondary, clearcoat effects, energy bound.
- Integration (when `bpy` is available): Principled property extraction; alpha CLIP culling; planar animation/centroid change.

Run: `pytest tests/lidar -q` (Blender tests auto‑skip without `bpy`).

### Related Research
- [Sensor-Realistic LiDAR Simulation](https://arxiv.org/pdf/2208.05961.pdf)
- [Simulating LiDAR Point Clouds for Autonomous Driving](https://arxiv.org/pdf/2008.08439.pdf)
- [LiDAR Intensity Calibration: A Review](https://www.mdpi.com/2072-4292/12/17/2697)
