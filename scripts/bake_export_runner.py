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


def _ensure_textures_dir(out_dir: Path, in_dir: Path) -> None:
    try:
        blends = [p for p in in_dir.iterdir() if p.suffix == ".blend"]
    except Exception:
        blends = []
    if not blends:
        # Default to scene.blend naming
        tgt = out_dir / "export_scene.blend" / "textures"
        tgt.mkdir(parents=True, exist_ok=True)
        return
    for b in blends:
        tgt = out_dir / f"export_{b.name}" / "textures"
        tgt.mkdir(parents=True, exist_ok=True)


def _clean(name: str) -> str:
    return name.replace(" ", "_").replace(".", "_")


def _find_principled(mat):
    try:
        if not (mat and getattr(mat, "use_nodes", False) and mat.node_tree):
            return None
        for n in mat.node_tree.nodes:
            if getattr(n, "type", "") == "BSDF_PRINCIPLED":
                return n
    except Exception:
        return None
    return None


def _write_sidecars(out_dir: Path) -> None:
    """Write per (object, material) sidecar JSONs next to baked textures.

    Fields: alpha_mode, alpha_clip, ior or specular, transmission_roughness.
    """
    try:
        import json

        import bpy  # type: ignore
    except Exception:
        return

    # Collect all textures directories created by exporter
    textures_dirs = [p for p in out_dir.rglob("textures") if p.is_dir()]
    if not textures_dirs:
        return

    # Build sidecar dicts keyed by (object, material)
    sidecars = []
    for obj in bpy.context.scene.objects:
        if getattr(obj, "type", "") != "MESH" or not obj.material_slots:
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue
            record = {
                "alpha_mode": str(getattr(mat, "blend_method", "BLEND")).upper(),
                "alpha_clip": float(getattr(mat, "alpha_threshold", 0.5) or 0.5),
                "transmission_roughness": 0.0,
            }
            bsdf = _find_principled(mat)
            if bsdf is not None:
                try:
                    record["ior"] = float(bsdf.inputs.get("IOR").default_value)
                except Exception:
                    try:
                        record["specular"] = float(
                            bsdf.inputs.get("Specular").default_value
                        )
                    except Exception:
                        pass
                try:
                    tr = bsdf.inputs.get("Transmission Roughness")
                    if tr is not None:
                        record["transmission_roughness"] = float(tr.default_value)
                except Exception:
                    pass
            # If neither ior nor specular found, set a safe specular default
            if ("ior" not in record) and ("specular" not in record):
                record["specular"] = 0.5

            sidecars.append((obj.name, mat.name, record))

    if not sidecars:
        return

    # Write sidecars into every textures dir we find
    for texdir in textures_dirs:
        for oname, mname, rec in sidecars:
            outp = texdir / f"{_clean(oname)}_{_clean(mname)}.json"
            try:
                with outp.open("w", encoding="utf-8") as fh:
                    json.dump(rec, fh, indent=2)
            except Exception:
                continue


def main(argv: list[str]) -> int:
    # Expect: runner.py -- <in_dir> <out_dir> <res>
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    if len(argv) < 3:
        print(
            "Usage: bake_export_runner.py -- <in_dir> <out_dir> <res>", file=sys.stderr
        )
        return 2
    in_dir = Path(argv[0]).resolve()
    out_dir = Path(argv[1]).resolve()
    res = str(argv[2])

    _ensure_vendor()
    _stub_exr()
    # Force safe Cycles settings (CPU) to avoid GPU-only failures in headless
    try:
        import bpy  # type: ignore

        bpy.context.scene.render.engine = "CYCLES"
        if hasattr(bpy.context.scene, "cycles"):
            bpy.context.scene.cycles.device = "CPU"
    except Exception:
        pass

    # Pre-create expected textures directories so image saves succeed
    try:
        _ensure_textures_dir(out_dir, in_dir)
    except Exception:
        pass

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
    # After exporter writes textures, emit sidecars alongside
    try:
        _write_sidecars(out_dir)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(
            f"[bake_export_runner] WARN: failed to write sidecars: {e}", file=sys.stderr
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
