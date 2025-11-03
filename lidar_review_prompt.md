This README-style prompt (include it with your project archive) outlines the current Infinigen + LiDAR setup, key files for review, and the pipeline from Indoors scene generation through exporter PBR texture baking and LiDAR point cloud generation.

## Repository Context

- **Repo**: `Infinigen` fork (`git@github.com:erik-d123/Infinigen.git`)
- **Indoors reference**: `References/Infinigen.md`
- **Documentation**:
- `docs/Installation.md` – minimal Blender-Python script install path (in use here)
- `docs/HelloRoom.md`, `docs/ConfiguringCameras.md`, `docs/ExportingToExternalFileFormats.md`, `docs/GroundTruthAnnotations.md`
- **Indoors core**:
  - `infinigen_examples/generate_indoors.py`
  - Configs under `infinigen_examples/configs_indoor/*.gin`
  - Constraint DSL in `infinigen_examples/constraints/home.py` (and peers)
  - Pipeline execution `infinigen/core/execute_tasks.py`, `infinigen/core/util/pipeline.py`
- **Exporter**:
  - `scripts/bake_export_textures.sh` (runner)
  - `scripts/bake_export_runner.py` (invoked inside Blender)
  - Texture outputs under `outputs/.../export/export_scene.blend/textures/`
- **LiDAR module**:
  - `README_LIDAR.md`
  - `lidar/intensity_model.py` – essential reflectivity model (Lambert + Schlick mixing, transmission residual, alpha semantics)
  - `lidar/lidar_raycast.py` – ray cast loop, alpha handling, optional secondaries
  - `lidar/material_sampler.py` – samples exporter PBR textures (Base Color, Roughness, Metallic, Transmission). NormalTS usage currently disabled due to tangent-synthesis issues on Blender > 4-gon geometry.
  - `lidar/lidar_generator.py` – orchestrates per-frame LiDAR generation (`lidar_config.json`, `lidar_calib.json`, `camview/`, PLY outputs)
  - Tests: `tests/lidar/*` – radiometry, Blender integration, I/O
- **Utility scripts**:
  - `scripts/run_indoors_module.py`, `scripts/run_indoors_local.sh` – run Indoors inside Blender headless, add `.blender_site`, stub OpenEXR/Imath
  - **Updated `infinigen/tools/blendscript_path_append.py`**: now appends both repo root and `.blender_site` to `sys.path`. Without this change, Blender launched via the helper couldn’t see the vendor packages (e.g., `trimesh`, `scikit-learn`). This was necessary to get the Blender-Python script setup working reliably both locally and on the cluster.

## Current Workflow (macOS; mirrored on cluster)

1. **Indoors generation**  
   ```bash
   scripts/run_indoors_local.sh -- \
     --seed 0 --task coarse \
     --output_folder outputs/indoors/local_coarse \
     -g fast_solve singleroom \
     -p compose_indoors.terrain_enabled=False \
        restrict_solving.restrict_parent_rooms='["DiningRoom"]'
   ```
   - Adds `.blender_site` to Blender’s `sys.path`, stubs OpenEXR/Imath (coarse stage doesn’t need them), runs Indoors as a module (fixes “relative import” issues).

2. **PBR texture bake**  
   ```bash
   bash scripts/bake_export_textures.sh \
     outputs/indoors/local_coarse \
     outputs/indoors/local_export \
     1024
   ```
   - Runs Blender → `scripts/bake_export_runner.py`, avoids import-time OpenEXR error, writes textures under `outputs/.../export/export_scene.blend/textures/`.

3. **LiDAR generation**  
   ```bash
   python -m infinigen.launch_blender \
     -m lidar.lidar_generator -- \
     outputs/indoors/local_coarse/scene.blend \
     --export-bake-dir outputs/indoors/local_export/export_scene.blend/textures \
     --output_dir outputs/indoors/local_lidar \
     --frames 1-2 --camera Camera --preset VLP-16
   ```
   - `.blender_site` appended via `blendscript_path_append.py` (updated)
   - `material_sampler` currently skips NormalTS (see “open questions”)
   - Outputs: `lidar_frame_*.ply`, `camview/`, `lidar_calib.json`, `trajectory.json`, `timestamps.txt`

4. **Viewer (optional)**  
   ```bash
   python lidar/lidar_viewer.py \
     outputs/indoors/local_lidar --frame 1 --color intensity
   ```
   Requires user Python packages `open3d`, `plyfile` (not Blender’s).

## Environment Notes

- **Blender**: 4.2.0 (macOS app). Blender’s `sys.path` includes `repo/.blender_site` at launch (updated `blendscript_path_append.py`).
- **Repo-level vendor**: `.blender_site` contains PyYAML, scikit-learn, pandas, psutil, geomdl, trimesh, shapely, networkx, imageio, python-fcl (if present).
- **Homebrew**: `fcl`, `imath`, `openexr` installed; needed for optional modules, though OpenEXR Python binding is stubbed for bake/Indoors to avoid build issues.

## Key LiDAR Behavior

- `lidar/intensity_model.py`: Lambert diffuse + Schlick specular, metallic mixing, roughness shaping, IOR preference for dielectrics, transmission reduces reflectivity and yields a pass-through residual, alpha semantics (CLIP vs coverage) handled, intensity = reflectivity / r^p with auto-expose option.
- `lidar/material_sampler.py`: In the current code NormalTS usage is disabled because Blender fails tangent computation on > quad geometry. Reviewer can consider safe re-enabling (e.g., pre-triangulate exporter output or provide fallback).
- `lidar/lidar_generator.py`: Writes PLY (ASCII default), `camview/` NPZ (K,T,HW), `lidar_calib.json`, `trajectory.json`, `timestamps.txt`, optional IMU/poses. Accepts presets (e.g., VLP-16) and frames/camera range.
- `tests/lidar/*`: Radiometry unit tests, Blender integration tests (normal map fallback, export-based sampling, alpha semantics), I/O checks.

## Reviewer Task Suggestions

### Indoors (for context)
- Review `infinigen_examples/generate_indoors.py` pipelines and constraints.
- Confirm how camera rigs and IMU packaging align with docs (`ConfiguringCameras.md`, `GroundTruthAnnotations.md`).

### Export Bakes
- Check `scripts/bake_export_textures.sh` → `scripts/bake_export_runner.py` to confirm no runtime baking occurs—LiDAR reuses exporter PBR maps only.
- Validate the baked texture directory structure (`.../export/export_scene.blend/textures/`) and naming conventions.

### LiDAR
- Examine `lidar/material_sampler.py`, `lidar/intensity_model.py`, `lidar/lidar_raycast.py`, `lidar/lidar_generator.py`, `lidar/lidar_config.py` for exporter bake reuse and reflectivity equations.
- Confirm `README_LIDAR.md` matches implementation, tests cover core radiometry, and PLY outputs include fields mentioned (intensity, reflectivity, range, mat_class, cos_incidence, transmittance, etc.).
- Note NormalTS currently skipped; reviewer may recommend safe reinstatement or call out limitations.

### Integration & Scripts
- `scripts/run_indoors_module.py`, `scripts/run_indoors_local.sh` ensure `.blender_site` on sys.path, stub OpenEXR/Imath, run module with proper semantics—check their robustness.
- `scripts/bake_export_textures.sh` now uses the new runner; confirm no remaining OpenEXR import-time issues.
- `lidar/lidar_viewer.py` for PLY visualization (open3d requirement).

### Tests & Docs
- `tests/lidar/*` – review coverage: radiometry, Blender integration (Alpha CLIP, fallback paths), PLY I/O.
- `README_LIDAR.md` – check that it aligns with code changes (e.g., NormalTS note).
- `docs/Installation.md` minimal install path (followed here).

### Open Questions (for reviewer input)
- Should exporter output be pre-triangulated to support NormalTS sampling reliably?
- Are there better solutions than stubbing OpenEXR/Imath (e.g., Mac ARM binary) if EXR processing is needed?
- Is symmetry of LiDAR intensity/reflectivity behavior matching real indoor sensors (Lambert normalization, cos incidence, distance exponent choices)?
- Do we want to symlink `export/textures` for convenience or keep pointing to `export/export_scene.blend/textures`?

## Example Pipeline Summary

1. **Generate Indoors** – `scripts/run_indoors_local.sh …`
2. **Bake PBR textures** – `bash scripts/bake_export_textures.sh …`
3. **Run LiDAR** – `python -m infinigen.launch_blender -m lidar.lidar_generator …`
4. *(Optional)* **View results** – `python lidar/lidar_viewer.py …`

The repo archive plus this prompt should give ChatGPT Pro all the reference points to evaluate:
- Proper use of exporter-baked PBR textures (no runtime baking)
- Material sampling/alpha semantics in LiDAR pipeline
- Integration with Indoors pipeline outputs (camview, IMU, PLY fields)
- Scripts and tests that enforce expected behavior.
