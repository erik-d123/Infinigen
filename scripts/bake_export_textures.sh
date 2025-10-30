#!/usr/bin/env bash
# Bake PBR textures (Albedo/Roughness/Metallic/Transmission/Normal) for a scene
# using Infinigen's exporter. This does NOT export geometry, only textures.
#
# Usage:
#   bash scripts/bake_export_textures.sh <input_blend_folder> <output_export_folder> [resolution]
# Example:
#   bash scripts/bake_export_textures.sh \
#     outputs/video_dynamic_indoor/6ded2d2f/coarse \
#     outputs/video_dynamic_indoor/6ded2d2f/export \
#     1024

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <input_blend_folder> <output_export_folder> [resolution]"
  exit 1
fi

IN_DIR="$1"
OUT_DIR="$2"
RES="${3:-1024}"

if [ ! -d "$IN_DIR" ]; then
  echo "[bake_export_textures] ERROR: input folder not found: $IN_DIR" >&2
  exit 1
fi
mkdir -p "$OUT_DIR"

echo "[bake_export_textures] Baking textures:"
echo "  input : $IN_DIR"
echo "  output: $OUT_DIR"
echo "  res   : $RES"

# Run exporter inside Blender via infinigen launcher (no extra Blender flags after --)
# Invoke exporter module via infinigen launcher (-m) so .blender_site is prepended to sys.path
# Run exporter via a Python expression so we fully control sys.argv and ensure .blender_site is importable
REPO_ROOT="$(pwd)"
python -m infinigen.launch_blender --background --python-expr "import os, sys; vendor=os.path.join('$REPO_ROOT','.blender_site'); \
    (sys.path.insert(0, vendor) if os.path.isdir(vendor) and vendor not in sys.path else None); \
    sys.argv=['Blender','--input_folder','$IN_DIR','--output_folder','$OUT_DIR','-f','usdc','-r','$RES']; \
    import infinigen.tools.export as ex; ex.main(ex.make_args())"

echo "[bake_export_textures] Done. Textures at: $OUT_DIR/textures"
