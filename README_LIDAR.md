# Infinigen Indoors LiDAR (Baked‑Only)

Indoor LiDAR ground truth for Infinigen scenes. This implementation is intentionally lightweight and material‑faithful: it samples only exporter‑baked PBR textures (no live node evaluation), uses a compact Lambert + Schlick model for reflectivity, and writes standard PLY + camview outputs.

**Design Goals**
- Respect Infinigen/Blender materials via exporter bakes
- Keep the reflectivity model compact, bounded, and fast
- Provide clear, reproducible outputs for training and evaluation

**What It Does**
- Samples per‑hit PBR values (Base Color, Roughness, Metallic, Transmission) from baked textures at UVs
- Computes reflectivity from Lambert + Schlick with metallic mixing and roughness shaping
- Applies alpha semantics once per surface (CLIP vs BLEND/HASHED)
- Emits a single optional pass‑through secondary for transmissive surfaces
- Writes PLY point clouds + camview intrinsics/extrinsics and calibration

## Materials & Baked Inputs
- Baked‑only semantics. No node‑graph evaluation at runtime; no runtime baking.
- Per‑hit inputs read from exporter textures: Base Color (RGB), Roughness (R), Metallic (R), Transmission (R).
- Alpha (coverage) source: Principled “Alpha” (if unlinked). Default opacity used when Alpha is unset. (Optional: add an “ALPHA” bake if you want per‑pixel coverage.)
- Incidence normal: geometric (smoothed) normal at the hit in the minimal mode.
- Bake locations: typically `export/export_scene.blend/textures/{object}_{DIFFUSE|ROUGHNESS|METAL|TRANSMISSION}.png` next to the scene.

## Alpha Semantics
- CLIP (alpha clip)
  - If coverage A < alpha_threshold → cull the hit entirely (no return)
  - If A ≥ threshold → keep the hit at full strength (no scaling)
- BLEND / HASHED
  - Never cull by threshold
  - Scale reflectivity and intensity by coverage A (0..1)
- Secondary returns: coverage is applied per surface along the path (primary and secondary surfaces each apply their rule)

## Reflectivity & Intensity (High‑Level)
- Fresnel (Schlick): F(c) with F0 from IOR/specular for dielectrics; metallic tints F0 by Base Color
- Specular: `R_spec = F(c) · (1 − roughness)^2` (dielectrics also scaled by Specular)
- Diffuse (Lambert, no 1/π): `R_diff = (1 − metallic) · (1 − F) · albedo_luma · cos_incidence`
- Transmission: `T_mat = (1 − metallic) · Transmission`; final `Reflectivity = (1 − T_mat) · clamp(R_spec + R_diff)`
- Intensity (pre‑exposure): `Intensity_raw = Reflectivity / distance^p` (p≈2)
- Alpha application at raycaster (once):
  - CLIP: cull when A < threshold; otherwise do not scale
  - BLEND/HASHED: never cull; scale by coverage A
- Secondary pass (optional): residual energy ∝ `T_mat · (1 − F) · (1 − trans_roughness)^2`; optional merge by range epsilon keeps the stronger return
- Exposure (8‑bit intensity only): percentile mapping per‑frame when enabled; reflectivity remains linear (float)

## CLI & Configuration (Quick Reference)
- Scene & camera
  - scene_path: .blend file for the scene
  - --camera: camera object name (defaults to scene.camera, or first camera)
  - --frames: single value, comma list, or inclusive range
- Baked textures (required)
  - --export-bake-dir: folder with baked textures; auto‑detected near the scene when present; else a clear error is raised
- Output framing & files
  - --output_dir: destination folder (auto‑created)
  - --ply-frame: `sensor` (default, +X forward, +Y left, +Z up), `camera`, or `world`
  - --ply-binary: binary PLY instead of ASCII
- Sensor sampling
  - --preset: `VLP-16`, `HDL-32E`, `HDL-64E`, `OS1-128`
  - --force-azimuth-steps: override azimuth samples per ring
- Radiometry & exposure
  - --secondary: enable single pass‑through return
  - --auto-expose: remap 95th percentile of positive intensities; reflectivity remains unscaled
  - --seed: seed for NumPy/random (geometry unchanged; ray pattern deterministic by preset)

## Outputs
- PLY per frame (ASCII or binary): xyz, intensity (U8), ring, azimuth, elevation, return_id, num_returns; optional: range_m, cos_incidence, mat_class, reflectivity (float), transmittance (float), normals
- camview npz: intrinsics K, extrinsics T, and resolution HW for consumers that expect Infinigen camview format
- lidar_calib.json: sensor_to_camera_R_cs, frame mode, and key sensor parameters
- trajectory.json and timestamps.txt: minimal trajectory and frame times

## Workflow & Integration
1) Generate Indoors scene(s)
2) Bake exporter textures once per scene (Base Color, Roughness, Metallic, Transmission)
3) Run LiDAR with the baked textures path (auto‑detected for common layouts)
4) Use the PLY/camview/calibration for downstream tasks

## Troubleshooting & Notes
- Missing textures folder: ensure the exporter ran; provide `--export-bake-dir` or re‑run the bake
- CLIP yields zero points when coverage < threshold; use BLEND/HASHED to scale by coverage
- Reflectivity is a linear float channel for training; intensity is 8‑bit for visualization/storage
- Minimal mode uses geometric normals only; NormalTS can be added later with triangulation at export

## Testing & Validation

- Runner
  - All tests run inside Blender (headless) via `scripts/run_lidar_tests.sh` which injects `.blender_site` and invokes pytest within Blender’s Python.

- Bake fixture
  - Tests first try the real exporter (tiny resolution) in a separate Blender process.
  - If textures are not produced (e.g., GPU/driver limits), tests synthesize tiny “fake bakes” that encode Principled defaults at the expected exporter filenames, preserving baked‑only semantics with stable CI behavior.

- Coverage (what the suite checks)
  - Intensity model (unit): energy bound, distance law (1/r^p), metallic Fresnel, IOR preference, alpha coverage reporting.
  - Materials (baked): albedo scaling requires re‑bake; baked property extraction at a hit; alpha semantics (BLEND scales, CLIP culls below threshold).
  - Kinematics (baked): tilt reduces intensity; moving the plane farther increases range (and does not brighten with auto‑expose off); animation across frames behaves plausibly.
  - Transmission (baked): transmissive near‑surface reflectivity is not higher than opaque; optional secondary or merge by range epsilon; nearest‑return comparison robust to ordering.
  - I/O: `process_frame` writes PLY, camview npz (K,T,HW), and `lidar_calib.json` with expected keys.

- How to run
  - Full suite: `bash scripts/run_lidar_tests.sh`
  - Or explicitly: `python -m infinigen.launch_blender -m infinigen.tools.blendscript_path_append --python scripts/run_lidar_pytest.py -- --factory-startup -q tests/lidar`

- Optional additions (if desired)
  - Auto‑exposure sanity (95th percentile mapping; reflectivity unchanged).
  - Grazing‑dropout behavior for shallow incidence.
  - UV orientation micro‑check (prevent vertical flips on texture sampling).
  - Binary PLY smoke test.
