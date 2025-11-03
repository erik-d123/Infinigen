#!/usr/bin/env bash
# Simple helper to run Indoors locally in Blender using module semantics and
# the repo-local vendor site (.blender_site). Works on macOS and Linux.
#
# Examples:
#   scripts/run_indoors_local.sh -- \
#     --seed 0 --task coarse \
#     --output_folder outputs/indoors/local_coarse \
#     -g fast_solve singleroom \
#     -p compose_indoors.terrain_enabled=False \
#        restrict_solving.restrict_parent_rooms='["DiningRoom"]'
#
# Notes:
# - Pass Indoors args after "--" so Blender forwards them to the module.
# - This script stubs OpenEXR/Imath in the Python runner; coarse does not need them.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Resolve Blender binary: prefer repo macOS app, otherwise system blender
if [[ -x "$ROOT_DIR/Blender.app/Contents/MacOS/Blender" ]]; then
  BLENDER_BIN="$ROOT_DIR/Blender.app/Contents/MacOS/Blender"
elif command -v blender >/dev/null 2>&1; then
  BLENDER_BIN="$(command -v blender)"
else
  echo "[run_indoors_local] ERROR: Could not find Blender. Install Blender 4.2.0 or place it under Blender.app" >&2
  exit 1
fi

echo "[run_indoors_local] Using Blender: $BLENDER_BIN"

"$BLENDER_BIN" -noaudio --background \
  --python "$ROOT_DIR/scripts/run_indoors_module.py" -- "$@"

