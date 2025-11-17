# Infinigen Indoors LiDAR (Baked‑First)

Indoor LiDAR ground truth for Infinigen scenes. This implementation is intentionally lightweight and material‑faithful: it samples exporter‑baked PBR textures (no live node evaluation), uses a compact Lambert + Schlick model for reflectivity, and writes standard PLY + camview outputs.

**Design Goals**
- Respect Infinigen/Blender materials via exporter bakes
- Keep the reflectivity model compact, bounded, and fast
- Provide clear, reproducible outputs for training and evaluation

**What It Does**
- Strict baked‑only: requires exporter textures and per‑material sidecars; no Principled/Material fallbacks at runtime.
- Samples per‑hit PBR values (Base Color, Roughness, Metallic, Transmission) from baked textures at UVs; coverage comes from DIFFUSE alpha or an optional COVERAGE bake.
- Reads alpha semantics and BRDF scalars (alpha_mode/clip, ior or specular, transmission_roughness) from a sidecar JSON per object/material.
- Computes reflectivity from Lambert + Schlick with metallic mixing, roughness shaping, and mild angle attenuation on the specular lobe.
- Applies alpha semantics once per surface (CLIP vs BLEND/HASHED)
- Emits a single optional pass‑through secondary for transmissive surfaces
- Writes PLY point clouds + camview intrinsics/extrinsics and calibration

## Materials & Baked Inputs
- Baked‑first semantics. No node‑graph evaluation at runtime; no runtime baking.
- Per‑hit inputs read from exporter textures: Base Color (RGB[A]), Roughness (R), Metallic (R), Transmission (R). Coverage comes from DIFFUSE alpha or an explicit COVERAGE (R) bake when present.
- Per‑material sidecar JSON placed next to textures provides: `alpha_mode`, `alpha_clip`, and either `ior` or `specular` (for Fresnel F0), plus optional `transmission_roughness`.
- Incidence normal: geometric (smoothed) normal at the hit in the minimal mode. (can add using normal bake (shading normal?))
- Bake locations: auto‑detected near the scene; e.g., `.../textures/{object}_{DIFFUSE|ROUGHNESS|METAL|TRANSMISSION}.png` and `{object}_{material}.json`.

## Alpha Semantics
- CLIP (alpha clip)
  - If coverage A < alpha_clip → cull the hit entirely (no return)
  - If A ≥ alpha_clip → keep the hit at full strength (no scaling)
- BLEND / HASHED
  - Never cull by threshold
  - Scale reflectivity and intensity by coverage A (0..1)
- Secondary returns: coverage is applied per surface along the path (primary and secondary surfaces each apply their rule)

Note: Alpha mode/threshold come from the sidecar JSON; coverage comes from DIFFUSE alpha or an explicit COVERAGE bake.

## Reflectivity & Intensity (High‑Level)
- Fresnel (Schlick): F(c) with F0 from IOR/specular for dielectrics; metallic tints F0 by Base Color
- Specular: `R_spec = F(c) · (1 − roughness)^2 · cos_incidence^k` (k≈0.5; dielectrics also scaled by Specular)
- Diffuse (Lambert, no 1/π): `R_diff = (1 − metallic) · (1 − F) · albedo_luma · cos_incidence`
- Transmission: `T_mat = (1 − metallic) · Transmission`; final `Reflectivity = (1 − T_mat) · clamp(R_spec + R_diff)`
- Intensity (pre‑exposure): `Intensity_raw = Reflectivity / distance^p` (p≈2)
- Alpha application at raycaster (once):
  - CLIP: cull when A < alpha_clip; otherwise do not scale
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
  - config (gin/dataclass): `specular_angle_power` controls mild specular angle attenuation (default 0.5)

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

## Current vs. Ideal Material Sources

This LiDAR path is baked‑first. We still read a few scalars and semantics from materials today, and there is a clean path to make everything strictly baked‑only.

- What still comes from Principled/materials (no node eval):
  - Alpha semantics: alpha_mode (CLIP/BLEND/HASHED) and alpha_clip threshold.
  - Dielectric Fresnel base: IOR (preferred) or Specular scalar when metallic=0.
  - Transmission roughness (optional scalar).
- No fallbacks at runtime: if a required baked map or sidecar field is missing, LiDAR errors. Bake/export must provide the signals.

- Why not “Principled only”:
  - Real materials are graphs; socket defaults are not the evaluated textures. Per‑hit node evaluation is too slow and non‑deterministic.
  - LiDAR needs PBR parameter maps (albedo/roughness/metallic/transmission) independent of lighting/view → baking is the robust path.

-- Required baked‑only pathway:
  - Coverage: use DIFFUSE alpha; optionally bake a dedicated COVERAGE map.
  - Sidecar semantics: write a tiny per‑material JSON next to textures with
    - alpha_mode: CLIP|BLEND|HASHED
    - alpha_clip: float in [0,1]
    - ior (or specular) and optional transmission_roughness
  - Filenames: `{object}_{SUFFIX}.png` for textures and `{object}_{material}.json` for sidecars.
  - Precedence: COVERAGE (if present) overrides DIFFUSE alpha for coverage.

### Example Sidecar

```
{
  "alpha_mode": "CLIP",
  "alpha_clip": 0.5,
  "ior": 1.45,
  "transmission_roughness": 0.0
}
```

This keeps runtime strictly baked‑only without touching Principled/material sockets.

## Technical Notes

- Radiometry model
  - Fresnel: Schlick F(cos_i) with metallic mixing; dielectrics prefer IOR → F0, else Specular.
  - Specular: `F * (1 - roughness)^2 * cos_i^k` (k≈0.5) and dielectrics scaled by Specular.
  - Diffuse: Lambert (no 1/π) `((1 - metallic) * (1 - F) * albedo_luma * cos_i)`.
  - Transmission: `T_mat = (1 - metallic) * Transmission`; reflectivity = `(1 - T_mat) * clamp(R_spec + R_diff)`.
  - Intensity: `reflectivity / distance^p` (p≈2), then optional U8 exposure mapping by percentile.
  - Secondary: single pass‑through using residual `T_mat * (1 - F) * (1 - trans_roughness)^2`, forward cast with bias, optional merge by range epsilon.

- Alpha semantics
  - CLIP: if coverage < alpha_clip → cull return; otherwise do not scale.
  - BLEND/HASHED: never cull; scale energy by coverage in [0,1].
  - Coverage source: DIFFUSE alpha or COVERAGE bake; alpha_mode/clip from sidecar.

- Transforms and frames
  - Sensor frame: +X forward, +Y left, +Z up. Blender camera: +X right, +Y up, −Z forward.
  - World→camera: `R_cw = R_wc^T`, `p_cam = R_cw * (p_world - t_wc)`.
  - Camera↔sensor rotation: see `sensor_to_camera_rotation()`; PLY can be written in sensor, camera, or world frames.

- Outputs & fields
  - PLY includes xyz, intensity(u8), ring(u16), azimuth, elevation, return_id, num_returns; optional range_m, cos_incidence, mat_class, reflectivity, transmittance.
  - Camview npz packs K (3×3), T (4×4), and HW; `lidar_calib.json` includes the sensor↔camera rotation and key sampling params.

- Performance & determinism
  - Baked texture sampling avoids per‑hit node evaluation; numpy bilinear sampling; no dependency on GPU shading.
  - Results are stable across Blender versions/hardware for the same scene assets and bakes.
