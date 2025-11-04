#!/usr/bin/env python3
"""
Run Infinigen Indoors as a Python module inside Blender, with local vendor site and
safe fallbacks for optional EXR modules. This avoids relative‑import issues and
lets you pass Indoors CLI args after a "--".

Usage (macOS example):

  BL="$(pwd)/Blender.app/Contents/MacOS/Blender"
  "$BL" -noaudio --background \
    --python scripts/run_indoors_module.py -- \
    --seed 0 --task coarse \
    --output_folder outputs/indoors/local_coarse \
    -g fast_solve singleroom \
    -p compose_indoors.terrain_enabled=False \
       restrict_solving.restrict_parent_rooms='["DiningRoom"]'

This script:
- Prepends repo local vendor path (.blender_site) to sys.path
- Stubs OpenEXR/Imath if unavailable (coarse stage does not require them)
- Runs infinigen_examples.generate_indoors as a proper module
"""

from __future__ import annotations

import os
import runpy
import sys
import types


def _ensure_vendor_in_path() -> None:
    root = os.getcwd()
    vendor = os.path.join(root, ".blender_site")
    if os.path.isdir(vendor) and vendor not in sys.path:
        sys.path.insert(0, vendor)


def _stub_optional_exr_modules() -> None:
    # Some rendering utilities import OpenEXR/Imath at module import time. Coarse generation
    # does not need them; stub so imports don’t crash on machines without EXR dev libs.
    try:
        import OpenEXR  # type: ignore  # noqa: F401
    except Exception:
        sys.modules["OpenEXR"] = types.ModuleType("OpenEXR")
    try:
        import Imath  # type: ignore  # noqa: F401
    except Exception:
        sys.modules["Imath"] = types.ModuleType("Imath")


def main() -> int:
    _ensure_vendor_in_path()
    _stub_optional_exr_modules()
    # Ensure repo root is importable as a package
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    # If Blender was invoked with "-- ..." after --python, strip the sentinel so
    # the Indoors argparse sees only its own flags.
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        sys.argv = [sys.argv[0]] + sys.argv[idx + 1 :]

    # Execute module with package semantics so relative imports work
    runpy.run_module("infinigen_examples.generate_indoors", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
