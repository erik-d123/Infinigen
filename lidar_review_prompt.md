# Infinigen LiDAR Review Prompt

I’ve attached `repo.zip`. Please review with this order:

1. Read `references/Infinigen.md` to absorb the Infinigen Indoors paper (procedural rooms, assets, constraint solver).
2. Study the indoor docs in `docs/` (`HelloRoom.md`, `ConfiguringInfinigen.md`, `ConfiguringCameras.md`, etc.) to understand how scenes are generated and cameras are configured.
3. Examine the indoor pipeline code: `infinigen_examples/generate_indoors.py` and its supporting utils/configs/constraints, so you see how the solver, assets, and camera rigs fit together.
4. Dive deep into the LiDAR implementation: `lidar/intensity_model.py`, `lidar_raycast.py`, `lidar_generator.py`, `lidar_config.py`, the tests under `tests/lidar/`, and `scripts/debug_lidar_checks.py`.

---

## LiDAR Ground-Truth Scope

I want physically grounded intensities that:

- Respond to Lambertian diffuse, metallic/specular (mirror-like), transmissive/frosted glass, and mixed Principled BSDF materials by sampling their inputs/textures.
- Use distance and incidence-angle falloff, with optional single pass-through secondary returns for transmissive surfaces.
- Do *not* yet cover advanced sensor noise, per-ring calibration, beam divergence, etc.—keep it lean but accurate.

You can think aloud and reason step-by-step (feel free to chain thoughts before summarizing). OpenAI Pro is fine with detailed chain-of-thought before the final answer.

---

## What to Deliver

Provide a focused code review on the LiDAR components:

1. **Intensity & Material Sampling:** Assess strengths, gaps, or missing edge cases (mirrors, stacked glass, animated materials).
2. **Performance/Maintenance:** Flag any pipeline issues in the generator/raycaster (e.g., caching, Blender depsgraph usage, scaling concerns).
3. **Testing:** Evaluate current coverage (Blender sanity tests, any headless/unit tests) and propose additions.
4. **Recommendations:** Suggest short-term fixes and longer-term improvements that keep the LiDAR pipeline physically credible while respecting the scoped feature set (no overkill noise modeling yet).

Feel free to explore intermediate reasoning before summarizing your findings.
