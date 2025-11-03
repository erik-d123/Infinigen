#!/usr/bin/env python3
"""
Lightweight Blender runner for exporter texture baking that:
- Adds repo vendor site (.blender_site) to sys.path
- Stubs OpenEXR/Imath (not required for baking)
- Invokes infinigen.tools.export with provided args

Usage (invoked by scripts/bake_export_textures.sh):
  Blender ... --python scripts/bake_export_runner.py -- <in_dir> <out_dir> <res>
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path


def _ensure_vendor() -> None:
    root = os.getcwd()
    vendor = os.path.join(root, ".blender_site")
    if os.path.isdir(vendor) and vendor not in sys.path:
        sys.path.insert(0, vendor)


def _stub_exr() -> None:
    try:
        import OpenEXR  # noqa: F401
    except Exception:
        sys.modules["OpenEXR"] = types.ModuleType("OpenEXR")
    try:
        import Imath  # noqa: F401
    except Exception:
        sys.modules["Imath"] = types.ModuleType("Imath")


def main(argv: list[str]) -> int:
    # Expect: runner.py -- <in_dir> <out_dir> <res>
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    if len(argv) < 3:
        print("Usage: bake_export_runner.py -- <in_dir> <out_dir> <res>", file=sys.stderr)
        return 2
    in_dir = Path(argv[0]).resolve()
    out_dir = Path(argv[1]).resolve()
    res = str(argv[2])

    _ensure_vendor()
    _stub_exr()

    # Build argv for exporter CLI
    sys.argv = [
        "Blender",
        "--input_folder",
        str(in_dir),
        "--output_folder",
        str(out_dir),
        "-f",
        "usdc",
        "-r",
        res,
    ]
    from infinigen.tools import export as ex

    ex.main(ex.make_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

