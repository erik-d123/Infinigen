#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover
    bpy = None


def bake_current_scene(tmp_dir: Path, res: int = 64) -> Path:
    """Save the current Blender scene to a temp input folder and run a tiny exporter bake.

    Returns the textures directory path produced by the exporter, which should be passed
    to LiDAR via cfg.export_bake_dir.
    """
    assert bpy is not None, "bake_current_scene must run inside Blender"

    tmp_dir = Path(tmp_dir)
    in_dir = tmp_dir / "coarse"
    out_dir = tmp_dir / "export"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    blend_path = in_dir / "scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    repo_root = Path(__file__).resolve().parents[2]
    runner = repo_root / "scripts" / "bake_export_runner.py"

    # Launch a separate Blender to run the exporter bake runner (stubs EXR and appends .blender_site)
    cmd = [
        sys.executable,
        "-m",
        "infinigen.launch_blender",
        "-m",
        "infinigen.tools.blendscript_path_append",
        "--",
        "--python",
        str(runner),
        "--",
        str(in_dir),
        str(out_dir),
        str(int(res)),
    ]
    subprocess.run(cmd, cwd=str(repo_root), check=True)

    # Resolve textures directory
    candidates = [
        out_dir / "export_scene.blend" / "textures",
        out_dir / "textures",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"No textures folder found under {out_dir}")

