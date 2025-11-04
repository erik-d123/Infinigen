#!/bin/bash
# Infinigen LiDAR Ground Truth Generator and Viewer (lean)

set -euo pipefail

# ---------- Defaults (override via CLI) ----------
SCENE_PATH="outputs/video_dynamic_indoor/6ded2d2f/coarse/scene.blend"
FRAMES="29-30"
CAMERA="Camera"

# Sensor preset (resolution determined by preset policy)
PRESET="OS1-128"   # VLP-16 | HDL-32E | HDL-64E | OS1-128

# Output frame for PLY (sensor recommended for LiDAR POV)
PLY_FRAME="sensor"     # sensor | camera | world

# Let user optionally supply: SCENE_PATH OUTPUT_DIR FRAMES CAMERA PRESET [FORCE_AZIMUTH_STEPS] [EXPORT_BAKE_DIR]
OUTPUT_DIR=""  # computed later unless provided
if [ $# -ge 1 ]; then SCENE_PATH="$1"; fi
if [ $# -ge 2 ]; then OUTPUT_DIR="$2"; fi
if [ $# -ge 3 ]; then FRAMES="$3"; fi
if [ $# -ge 4 ]; then CAMERA="$4"; fi
if [ $# -ge 5 ]; then PRESET="$5"; fi
# Advanced option: FORCE_AZIMUTH (not typically needed)
FORCE_AZIMUTH_STEPS=""
if [ $# -ge 6 ]; then FORCE_AZIMUTH_STEPS="$6"; fi
EXPORT_BAKE_DIR_ARG=""
if [ $# -ge 7 ]; then EXPORT_BAKE_DIR_ARG="$7"; fi


# Build default OUTPUT_DIR (after any overrides) if not provided
if [ -z "${OUTPUT_DIR}" ]; then
  SCENE_NAME="$(basename "$(dirname "$(dirname "$SCENE_PATH")")")"
  TS="$(date +"%Y%m%d_%H%M%S")"
  PRESET_LOWER="$(echo "$PRESET" | tr '[:upper:]' '[:lower:]')"
  OUTPUT_DIR="outputs/${PRESET_LOWER}/${SCENE_NAME}/${TS}"
fi
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Infinigen LiDAR Generator"
echo "Scene:   $SCENE_PATH"
echo "Camera:  $CAMERA"
echo "Frames:  $FRAMES"
echo "Preset:  $PRESET"
echo "PLY:     ascii, frame=${PLY_FRAME}"
echo "Output:  $OUTPUT_DIR"
echo "========================================"

echo "Generating LiDAR data..."

# Determine export-bake dir if not provided: search common exporter layouts
RESOLVED_BAKE_DIR=""
if [ -n "$EXPORT_BAKE_DIR_ARG" ]; then
  RESOLVED_BAKE_DIR="$EXPORT_BAKE_DIR_ARG"
else
  SCENE_DIR="$(dirname "$SCENE_PATH")"
  SCENE_NAME="$(basename "$SCENE_PATH")"
  CANDIDATES=(
    "$SCENE_DIR/export_${SCENE_NAME}/textures"
    "$SCENE_DIR/export_scene.blend/textures"
    "$(dirname "$SCENE_DIR")/export/export_${SCENE_NAME}/textures"
    "$(dirname "$SCENE_DIR")/export/textures"
  )
  for C in "${CANDIDATES[@]}"; do
    if [ -d "$C" ]; then RESOLVED_BAKE_DIR="$C"; break; fi
  done
fi

if [ -z "$RESOLVED_BAKE_DIR" ]; then
  echo "[warn] Could not auto-detect export-bake textures. Provide as 7th arg or run scripts/bake_export_textures.sh." >&2
else
  echo "Using export-bake textures: $RESOLVED_BAKE_DIR"
fi
python -m infinigen.launch_blender -m lidar.lidar_generator -- \
  "$SCENE_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --frames "$FRAMES" \
  --camera "$CAMERA" \
  --preset "$PRESET" \
  $([ -n "$RESOLVED_BAKE_DIR" ] && echo "--export-bake-dir $RESOLVED_BAKE_DIR") \
  $([ -n "$FORCE_AZIMUTH_STEPS" ] && echo "--force-azimuth-steps $FORCE_AZIMUTH_STEPS") \
  --ply-frame "$PLY_FRAME" \
  --seed 0

echo "Data generation complete!"
echo "Saved to: $OUTPUT_DIR"
echo
echo "View with:"
echo "  python lidar/lidar_viewer.py \"$OUTPUT_DIR\" --color intensity_heat           # Intensity coloring"
echo "  python lidar/lidar_viewer.py \"$OUTPUT_DIR\" --color ring                 # Ring (elevation) coloring"
echo "  python lidar/lidar_viewer.py \"$OUTPUT_DIR\" --camera-view                # LiDAR POV (uses poses_tum.txt when present)"
echo "========================================"
