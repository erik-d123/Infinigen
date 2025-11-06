# Infinigen + Indoors + LiDAR — Deep Context Prompt for LLM Agents

This document onboards an LLM agent to the Infinigen codebase with a focus on the Indoors scene‑generation pipeline and the LiDAR ground‑truth implementation. It enumerates the most relevant files, suggests a reading order, explains key abstractions and data contracts, and provides execution tips so the agent can be productive quickly and safely.

## Objective

- Understand how Infinigen Indoors composes, constrains, and solves indoor scenes.
- Understand the LiDAR ground‑truth generator: its baked‑only material sampling, radiometric model, alpha semantics, transforms, and outputs.
- Be able to navigate, debug, and extend features without changing behavior unintentionally.

## Use Judgment and Expand Scope

This brief lists high‑value files, but it is not exhaustive. As you reason:
- Follow imports to adjacent modules when a public API is referenced but details matter.
- Prefer reading source over inferring behavior from names.
- If you see configuration indirection (e.g., gin, env flags), locate the config or default definitions.
- Treat docs/references as hints; defer to code for truth. When in doubt, search the repository and broaden your context.

Recommended tactics:
- Use fast, broad search (e.g., ripgrep) for keywords, symbols, class names, and tags.
- Skim module headers to map responsibilities; dive deeper when a component affects your question.
- When a file path referenced here is missing, search for similarly named modules or relocated code.

## High‑Level Repo Map (selected; non‑exhaustive)

- Indoors entry and configs
  - `infinigen_examples/generate_indoors.py`
  - `infinigen_examples/util/generate_indoors_util.py`
  - `infinigen_examples/constraints/home.py`
  - `infinigen_examples/configs_indoor/base_indoors.gin`
  - `infinigen_examples/configs_indoor/fast_solve.gin`
  - `infinigen_examples/configs_indoor/singleroom.gin`
  - `infinigen_examples/configs_indoor/overhead.gin`
  - `infinigen_examples/configs_indoor/real_geometry.gin`
  - `infinigen_examples/configs_indoor/multistory.gin`
  - `scripts/run_indoors_module.py`
  - `scripts/run_indoors_local.sh`
  - `scripts/indoor.sh`

- Constraint language & solver (rooms, greedy object placement)
  - `infinigen/core/constraints/constraint_language/*` (API surface; referenced)
  - `infinigen/core/constraints/example_solver/solve.py`
  - `infinigen/core/constraints/example_solver/annealing.py`
  - `infinigen/core/constraints/example_solver/propose_*.*`
  - `infinigen/core/constraints/example_solver/greedy/*`
  - `infinigen/core/constraints/example_solver/room/solver.py`
  - `infinigen/core/constraints/example_solver/room/decorate.py`
  - `infinigen/core/constraints/example_solver/room/*`
  - `infinigen/core/constraints/evaluator/indoor_util.py`

- Scene utilities & pipeline
  - `infinigen/core/util/pipeline.py`
  - `infinigen/core/util/camera.py`
  - `infinigen/core/util/blender.py`
  - `infinigen/core/util/ocmesher_utils.py` (invoked conditionally)
  - `infinigen/terrain/*` (coarse terrain backdrop integration)

- LiDAR implementation
  - `README_LIDAR.md`
  - `generate_and_view_lidar.sh`
  - `lidar/lidar_generator.py`
  - `lidar/lidar_raycast.py`
  - `lidar/intensity_model.py`
  - `lidar/material_sampler.py`
  - `lidar/mesh_uv.py`
  - `lidar/lidar_io.py`
  - `lidar/lidar_config.py`
  - `lidar/lidar_scene.py`
  - `lidar/lidar_viewer.py`

- References & general docs
  - `References/Infinigen.md`
  - `README.md`

Note: Some referenced “docs/*.md” files in `README.md` may not exist locally in this workspace; they’re part of upstream documentation. Use `References/Infinigen.md` and in‑repo source for ground truth. If additional docs or design notes are present elsewhere, include them in your review.

---

## Indoors Pipeline — Concepts & Flow

Indoors composes a multi‑room interior and populates it using a constraint language plus a greedy/annealed solver.

Core flow (see `infinigen_examples/generate_indoors.py`):

1) Config & initialization
   - Uses gin (`*.gin`) to configure solver steps, camera behavior, lighting, etc.
   - `compose_indoors(...)` is the main pipeline entry.

2) Terrain & lighting (optional for backdrop)
   - Coarse terrain may be added to form outdoor context.
   - Sky lighting configured.

3) Constraint graph construction
   - Room grammar & layout constraints: `infinigen_examples/constraints/home.py` → `home_room_constraints()` returns a `cl.Problem` for floor plan generation.
   - Object placement constraints: `home_furniture_constraints()` encodes semantics for furniture categories, coverage, accessibility, and room‑type specifics (kitchens, bathrooms, etc.).

4) Greedy stages & solver
   - `default_greedy_stages()` defines domains (rooms, floor/wall/ceiling, side objects, on‑top/on‑support) determining solve order.
   - `Solver.solve_rooms(...)` builds a `State` via `FloorPlanSolver` with room polygons across stories.
   - `Solver.solve_objects(...)` performs simulated annealing moves (addition / deletion / relation plane change / resample / reinit pose / translate / rotate) over active domains.

5) Camera posing/animation
   - Cameras are spawned and base views computed based on solved geometry; optional animation and IMU export.

6) Asset population & room decoration
   - Placeholder population, then actual door/window/stair assets, skirting boards.
   - Room split into wall/floor/ceiling meshes and materials assigned per room type (`room/decorate.py`).

7) Optional occlusion meshing
   - `enable_ocmesh_room` and `convert_shader_displacement(...)` adjust room meshes for real geometry needs.

8) Backdrop & final tweaks
   - `generate_indoors_util.create_outdoor_backdrop(...)` prunes terrain in house bbox and adds clouds/grass/rocks as configured.

Key solver internals to scan:
- `infinigen/core/constraints/example_solver/solve.py`
- `infinigen/core/constraints/example_solver/annealing.py`
- `infinigen/core/constraints/example_solver/room/solver.py`
- `infinigen/core/constraints/example_solver/propose_*.*`

Important configuration knobs:
- `infinigen_examples/configs_indoor/base_indoors.gin` (solve steps, temps, camera params)
- `fast_solve.gin` (reduced iterations), `singleroom.gin`, `overhead.gin`, `real_geometry.gin`, `multistory.gin`

Useful scripts:
- `scripts/run_indoors_module.py`: runs Indoors as a proper Python module under Blender; stubs optional EXR libs.
- `scripts/run_indoors_local.sh`: convenience wrapper for local Blender runs.

---

## LiDAR Ground Truth — Design & Flow

Design goals (see `README_LIDAR.md`): baked‑only semantics for PBR inputs, compact Lambert + Schlick reflectivity, alpha semantics applied once, optional single pass‑through secondary, and standard PLY + camview outputs.

Execution path (see `lidar/lidar_generator.py`):
1) Open scene or use current Blender context; resolve camera and frames.
2) Require exporter‑baked textures directory; auto‑detect common layouts near the scene folder.
3) Generate sensor rays (by preset or override) in sensor frame (+X forward, +Y left, +Z up).
4) Transform rays to world via camera pose, then `scene.ray_cast(...)` against evaluated depsgraph.
5) For each hit:
   - Sample material properties at UV using baked textures: Base Color, Roughness, Metallic, Transmission (`material_sampler.py`).
   - Fall back to unlinked Principled defaults for missing channels; read alpha semantics (blend mode, threshold) and IOR/specular when needed (`intensity_model.py`).
   - Compute per‑hit reflectivity: Schlick Fresnel with metallic mixing, roughness shaping, lambertian diffuse (no 1/π), transmission shaping.
   - Apply alpha semantics once: CLIP culls below threshold; BLEND/HASHED scales energy by coverage.
   - Optionally spawn one secondary (pass‑through) hit using residual energy; merge by range epsilon if appropriate (`lidar_raycast.py`).
6) Write PLY in chosen frame (sensor/camera/world) and save camview intrinsics/extrinsics; write minimal calibration JSON.

Outputs & contracts (see `lidar/lidar_io.py`):
- PLY fields (base): x,y,z, intensity(u8), ring(u16), azimuth(f4), elevation(f4), return_id(u8), num_returns(u8)
- Optionals: range_m(f4), cos_incidence(f4), mat_class(u8), reflectivity(f4), transmittance(f4), normals
- Camview npz (per frame): K (3×3), T (4×4), HW
- Calibration JSON: frame_mode, sensor_to_camera_R_cs, min/max range, azimuth steps, rings

Coordinate frames:
- Sensor: +X forward, +Y left, +Z up (`lidar/lidar_scene.py: sensor_to_camera_rotation()`)
- Blender camera: +X right, +Y up, −Z forward
- Transform logic for PLY: world→camera, then camera→sensor when requested (`lidar/lidar_generator.py`)

Important LiDAR parameters (see `lidar/lidar_config.py`):
- Presets: VLP‑16, HDL‑32E, HDL‑64E, OS1‑128 (ring counts, ranges)
- Radiometry: distance_power, auto_expose target, global scale
- Secondary: enable, residual threshold, bias, min cos, merge epsilon
- Baked material sampling: `export_bake_dir` (required)

Viewer:
- `lidar/lidar_viewer.py` loads PLY attributes (prefers `plyfile`, falls back to ASCII reader or Open3D) and visualizes with color modes (intensity, heat, reflectivity, ring).

Convenience script:
- `generate_and_view_lidar.sh` runs LiDAR on a scene and launches the viewer; auto‑detects bake texture folder.

---

## Files To Read (Indoors)

Start with these to build mental models:

- Pipeline entry & stages
  - `infinigen_examples/generate_indoors.py`
  - `infinigen_examples/util/generate_indoors_util.py`

- Constraint programs
  - `infinigen_examples/constraints/home.py`
  - `infinigen_examples/constraints/semantics.py` (if present; semantics aliases)

- Solver internals
  - `infinigen/core/constraints/example_solver/solve.py`
  - `infinigen/core/constraints/example_solver/annealing.py`
  - `infinigen/core/constraints/example_solver/room/solver.py`
  - `infinigen/core/constraints/example_solver/propose_discrete.py`
  - `infinigen/core/constraints/example_solver/propose_continous.py`
  - `infinigen/core/constraints/evaluator/indoor_util.py`

- Room materials & post‑processing
  - `infinigen/core/constraints/example_solver/room/decorate.py`

- Configs & run scripts
  - `infinigen_examples/configs_indoor/*.gin`
  - `scripts/run_indoors_module.py`
  - `scripts/run_indoors_local.sh`

These lists are starting points, not limits. Expand them as needed.

## Files To Read (LiDAR)

- Design & usage
  - `README_LIDAR.md`
  - `generate_and_view_lidar.sh`

- Core modules
  - `lidar/lidar_generator.py`
  - `lidar/lidar_raycast.py`
  - `lidar/intensity_model.py`
  - `lidar/material_sampler.py`
  - `lidar/mesh_uv.py`
  - `lidar/lidar_io.py`
  - `lidar/lidar_scene.py`
  - `lidar/lidar_config.py`
  - `lidar/lidar_viewer.py`

## External Docs & References

- `References/Infinigen.md` — publication excerpt detailing Indoors contributions (constraint language, solver, export path).
- `README.md` — top‑level documentation and links; note that some `docs/*.md` files referenced may be absent locally.

---

## Suggested Reading Order

1) Publication + top‑level readme
   - `References/Infinigen.md`, `README.md`

2) Indoors pipeline
   - `infinigen_examples/generate_indoors.py` → stages & solver calls
   - `infinigen_examples/constraints/home.py` → constraint graph; soft/hard terms
   - `infinigen/core/constraints/example_solver/*` → annealing, greedy, room solver
   - `infinigen_examples/util/generate_indoors_util.py` → cameras, backdrops
   - `infinigen/core/constraints/example_solver/room/decorate.py` → materials/room splitting
   - `infinigen_examples/configs_indoor/*.gin` → parameters

3) LiDAR pipeline
   - `README_LIDAR.md` → design goals & outputs
   - `lidar/lidar_generator.py` → CLI, transforms, files
   - `lidar/lidar_raycast.py` → casting loop, alpha semantics, secondary
   - `lidar/intensity_model.py` → reflectivity model, baked sampling, IOR/specular
   - `lidar/material_sampler.py`, `lidar/mesh_uv.py` → UV sampling
   - `lidar/lidar_io.py` → PLY contract; `lidar/lidar_viewer.py` → viewing

---

## Style & Conventions (observed)

- Python style: concise functions, minimal but purposeful docstrings, type hints used pragmatically; logging over prints.
- Config via gin for Indoors; CLI via argparse for LiDAR.
- No runtime evaluation of shader nodes in LiDAR; prefer baked textures; Principled values read only for defaults/metadata (alpha mode/threshold, IOR/specular).
- Keep transforms explicit; sensor↔camera rotation defined in one place (`lidar/lidar_scene.py`).

---

## Execution Tips

- Indoors (module mode under Blender)
  - `scripts/run_indoors_local.sh -- --seed 0 --task coarse --output_folder outputs/indoors/local_coarse -g fast_solve singleroom`
  - Adjust with `-p` overrides, e.g. restrict rooms, disable terrain.

- LiDAR generation (under Blender’s Python)
  - `python -m infinigen.launch_blender -m lidar.lidar_generator -- path/to/scene.blend --output_dir outputs/lidar --frames 1-10 --camera Camera --preset OS1-128 --export-bake-dir <textures_dir>`
  - View: `python lidar/lidar_viewer.py outputs/lidar --color intensity_heat`

---

## Exploration Checklist (expand as needed)

- Entry points and orchestration
  - Identify CLI/module boundaries for Indoors and LiDAR; confirm argument parsing and defaulting.
- Constraint and solver paths
  - Trace from constraint graph construction to solver moves; note where hard/soft terms are evaluated.
- Geometry and transforms
  - Verify frame conventions (sensor, camera, world) and matrix assembly at write time.
- Material signals
  - Confirm which PBR channels are sampled from bakes; identify remaining metadata read from materials; check edge cases (multi‑material meshes, missing maps).
- Outputs & I/O contracts
  - Enumerate required/optional fields in PLY and camview; verify data types and shapes.
- Config surface
  - Survey gin configs affecting Indoors; list toggles that change solve behavior or camera placement.
- Performance & robustness
  - Look for safeguards (timeouts, iteration caps, fallbacks) and where they trigger.

## Non‑Goals & Assumptions

- The LiDAR path does not attempt node‑graph evaluation; it is “baked‑first” by design.
- Alpha is not currently baked as a texture; semantics are read from material settings and/or Principled Alpha default.
- This workspace may not include all upstream docs; rely on local sources and references enumerated here.

---

## Self‑Check Questions (for the agent)

1) Can you explain the order and purpose of Indoors greedy stages and how they map to solver moves?
2) How does `home_furniture_constraints()` encode fullness, accessibility, and kitchen‑specific rules?
3) What are the exact differences between CLIP vs BLEND/HASHED alpha semantics in LiDAR, and where are they applied?
4) How is reflectivity computed for dielectrics vs metals? Where does IOR/specular come from?
5) What fields are guaranteed in the LiDAR PLY, and which are optional?
6) How are sensor→world and world→sensor transforms assembled at write time?

---

## Extension Ideas (if asked to implement)

- Strict baked‑only mode: bake alpha (coverage) and per‑material sidecars for IOR/F0; remove Principled fallbacks.
- Additional LiDAR presets; variable azimuth step density per ring.
- Extended outputs: per‑hit normals, BRDF class confidences.
- Additional Indoors constraints (e.g., garages, offices) to broaden coverage.

---

## Quick Commands (Cheatsheet)

## When Answering Follow‑Up Questions

- Cross‑reference claims with specific files and line ranges you inspected.
- Be explicit about assumptions and where behavior is inferred vs. confirmed.
- If a concept depends on configuration, note the relevant gin/CLI flags and defaults.
- Propose simple validation steps or commands the user can run to confirm behavior.

- Indoors (coarse, single room):
  - `scripts/run_indoors_local.sh -- --seed 0 --task coarse --output_folder outputs/indoors/local -- -g fast_solve singleroom`

- LiDAR (sensor frame, OS1‑128):
  - `python -m infinigen.launch_blender -m lidar.lidar_generator -- outputs/indoors/local/coarse/scene.blend --output_dir outputs/os1/sample --frames 1-5 --camera Camera --preset OS1-128 --export-bake-dir <textures_dir>`

- View:
  - `python lidar/lidar_viewer.py outputs/os1/sample --color reflectivity`
