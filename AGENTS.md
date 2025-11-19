# Repository Guidelines

## Project Structure & Modules
- Core library in `infinigen/`; runnable drivers and configs in `infinigen_examples/` (e.g., `generate_indoors`, `generate_nature`).
- Tests in `tests/` with markers: `unit`, `blender`, `e2e`, `slow`, and `tests/lidar/` for LiDAR.
- Helper scripts in `scripts/` (e.g., `run_indoors_local.sh`, `run_lidar_tests.sh`, `debug_lidar_checks.py`). Outputs in `outputs/`. Vendor site-packages in `.blender_site/`.

## Indoors & LiDAR Workflow
- Generate an indoor scene (coarse):
  `scripts/run_indoors_local.sh -- --seed 0 --task coarse --output_folder outputs/indoors/local`
  This produces `.../coarse/scene.blend` used by LiDAR.
- Emit LiDAR ground truth from a `.blend`:
  `python -m infinigen.launch_blender -m infinigen.lidar.lidar_generator -- outputs/.../coarse/scene.blend --output_dir outputs/lidar/demo --frames 1-10 --camera Camera --preset VLP-16 --auto-expose --ply-frame sensor`
- Quick helper: `./generate_and_view_lidar.sh [SCENE_PATH] [OUTPUT_DIR] [FRAMES] [CAMERA] [PRESET]` then view with `python infinigen/lidar/lidar_viewer.py outputs/lidar/demo`.
- For OpenGL ground-truth features (non-LiDAR), build `customgt`: `make customgt`.

## Build, Test, and Dev
- Dev install (Python 3.11): `pip install -e .[dev]` (optionals: `[terrain]`, `[vis]`, `[wandb]`). Initialize submodules: `git submodule update --init --recursive`.
- Extras: `make terrain`, `make customgt`. Env toggles: `INFINIGEN_INSTALL_TERRAIN=True`, `INFINIGEN_INSTALL_CUSTOMGT=True`, `INFINIGEN_MINIMAL_INSTALL=True`.
- Lint/format: `ruff check .`, `ruff format .` or `pre-commit run -a`. Blender vendor deps: see `scripts/install_blender_vendor.sh` and `.blender_site/`.
- Tests: CPU-only `pytest -m "unit and not slow"`. Blender/LiDAR: `scripts/run_lidar_tests.sh` or `python -m infinigen.launch_blender --python scripts/run_lidar_pytest.py -- -m blender`.

## Style & Conventions
- Python: PEP 8, 4-space indent, type hints where practical. Modules `snake_case`, classes `CapWords`, constants `UPPER_SNAKE_CASE`. Prefer absolute imports; relative only for siblings (ruff `TID252`). C/C++/CUDA follow `.clang-format`.

## Commits & PRs
- Use imperative, scoped messages (e.g., `feat(indoors): add dining solver`, `fix(lidar): clamp intensity`).
- PRs include description, linked issues, validation (commands, log snippets, `outputs/...` paths), and doc/config updates.

## References
- Review `Publication.md` and any `references/` materials if present. Align LiDAR assumptions (sensor presets, radiometry) and Indoors configs with the cited works.
