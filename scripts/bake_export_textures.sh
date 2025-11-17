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
# If ENV_PATH is set by the caller, expose its site-packages to Blender as well
BLENDER_CONDA_SITE="${BLENDER_CONDA_SITE:-${ENV_PATH:-}/lib/python3.11/site-packages}"
python -m infinigen.launch_blender -m infinigen.tools.blendscript_path_append -- \
  --python "$(pwd)/scripts/bake_export_runner.py" -- "$IN_DIR" "$OUT_DIR" "$RES"

# Print the actual textures directory discovered under OUT_DIR
REAL_TEXDIR=$(python - "$OUT_DIR" <<'PY'
import sys, pathlib
out=pathlib.Path(sys.argv[1])
cands=[p for p in out.rglob('textures') if p.is_dir()]
print(str(cands[0] if cands else (out/'textures')))
PY
)
echo "[bake_export_textures] Done. Textures at: $REAL_TEXDIR"
