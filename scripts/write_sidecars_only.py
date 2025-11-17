#!/usr/bin/env python3
"""
Write LiDAR sidecar JSONs (alpha_mode/clip, IOR or Specular, transmission_roughness)
next to an existing textures directory, without re-baking.

Usage (run inside Blender via launcher):

  python -m infinigen.launch_blender -m infinigen.tools.blendscript_path_append -- \
    --python scripts/write_sidecars_only.py -- \
    /path/to/scene.blend /path/to/export/.../textures

Notes:
- This does not bake textures; it only emits sidecars based on current material settings.
- JSON filenames: {clean_object_name}_{clean_material_name}.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _clean(name: str) -> str:
    return name.replace(" ", "_").replace(".", "_")


def _find_principled(mat):
    try:
        if not (mat and getattr(mat, "use_nodes", False) and mat.node_tree):
            return None
        for node in mat.node_tree.nodes:
            if getattr(node, "type", "") == "BSDF_PRINCIPLED":
                return node
    except Exception:
        return None
    return None


def main(argv: list[str]) -> int:
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    if len(argv) < 2:
        print(
            "Usage: write_sidecars_only.py -- <scene.blend> <textures_dir>",
            file=sys.stderr,
        )
        return 2

    scene_path = Path(argv[0]).resolve()
    tex_dir = Path(argv[1]).resolve()
    tex_dir.mkdir(parents=True, exist_ok=True)

    # Load scene
    import bpy  # type: ignore

    bpy.ops.wm.open_mainfile(filepath=str(scene_path))

    count = 0
    for obj in bpy.context.scene.objects:
        if getattr(obj, "type", "") != "MESH" or not obj.material_slots:
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue
            rec = {
                "alpha_mode": str(getattr(mat, "blend_method", "BLEND")).upper(),
                "alpha_clip": float(getattr(mat, "alpha_threshold", 0.5) or 0.5),
                "transmission_roughness": 0.0,
            }
            bsdf = _find_principled(mat)
            if bsdf is not None:
                try:
                    rec["ior"] = float(bsdf.inputs.get("IOR").default_value)
                except Exception:
                    try:
                        rec["specular"] = float(
                            bsdf.inputs.get("Specular").default_value
                        )
                    except Exception:
                        pass
                try:
                    tr = bsdf.inputs.get("Transmission Roughness")
                    if tr is not None:
                        rec["transmission_roughness"] = float(tr.default_value)
                except Exception:
                    pass
            if "ior" not in rec and "specular" not in rec:
                rec["specular"] = 0.5

            outp = (
                tex_dir
                / f"{_clean(obj.name)}_{_clean(getattr(mat, 'name', 'Material'))}.json"
            )
            try:
                with outp.open("w", encoding="utf-8") as fh:
                    json.dump(rec, fh, indent=2)
                count += 1
            except Exception as e:
                print(
                    f"[write_sidecars_only] WARN: failed to write {outp}: {e}",
                    file=sys.stderr,
                )

    print(f"[write_sidecars_only] Wrote {count} sidecars into {tex_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
