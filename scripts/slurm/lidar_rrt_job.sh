#!/usr/bin/env bash
# Slurm job body for indoor RRT video + LiDAR pipeline
set -euo pipefail

echo "[lidar_rrt_job] Host: $(hostname)  User: $USER"
echo "[lidar_rrt_job] Output: ${JOB_OUTPUT_FOLDER:-unset}  Scenes: ${JOB_NUM_SCENES:-unset}  Frames: ${JOB_FRAME_RANGE:-unset}  Seed: ${JOB_SEED:-unset}"

# Activate environment
source ~/.bashrc || true
if conda env list | grep -q "${JOB_CONDA_ENV:-infinigen}"; then
  conda activate "${JOB_CONDA_ENV:-infinigen}"
elif command -v mamba >/dev/null 2>&1 && mamba env list | grep -q "${JOB_CONDA_ENV:-infinigen}"; then
  mamba activate "${JOB_CONDA_ENV:-infinigen}"
else
  echo "[lidar_rrt_job] WARNING: could not activate conda env ${JOB_CONDA_ENV:-infinigen}" >&2
fi
echo "[lidar_rrt_job] Conda env: $CONDA_DEFAULT_ENV"

# Ensure exporter deps are visible to Blender if needed
if [ -d ".blender_site" ]; then
  export PYTHONPATH="$(pwd)/.blender_site:${PYTHONPATH:-}"
fi

OUT_DIR="${JOB_OUTPUT_FOLDER:-outputs/video_dynamic_indoor}"
SCENES="${JOB_NUM_SCENES:-1}"
FRAMES="${JOB_FRAME_RANGE:-1-200}"
SEED="${JOB_SEED:-0}"
OCMESH="${JOB_ENABLE_OCMESH:-false}"
NO_BAKE_N="${JOB_NO_BAKE_NORMALS:-false}"

mkdir -p "$OUT_DIR"

PIPELINE_CFGS=( local_256GB.gin indoor_background_configs.gin blender_gt.gin )
GT_CFGS=( lidar_bake_textures.gin lidar.gin )

OVERRIDES=(
  compute_base_views.min_candidates_ratio=2
  compose_indoors.terrain_enabled=False
  compose_indoors.restrict_single_supported_roomtype=True
)

if [ "$OCMESH" = "true" ]; then
  OVERRIDES+=( compose_indoors.enable_ocmesh_room=True )
fi

PIPELINE_OVERRIDES=(
  get_cmd.driver_script='infinigen_examples.generate_indoors'
  iterate_scene_tasks.frame_range="[$(echo "$FRAMES" | sed 's/-/,/')]"
)

LIDAR_OVERRIDES=()
if [ "$NO_BAKE_N" = "true" ]; then
  # Disable normal usage inside LiDAR
  LIDAR_OVERRIDES+=( "lidar.lidar_config.LidarConfig.use_baked_normals=False" )
fi

set -x
python -m infinigen.datagen.manage_jobs \
  --output_folder "$OUT_DIR" \
  --num_scenes "$SCENES" \
  --specific_seed "$SEED" \
  --configs singleroom.gin rrt_cam_indoors.gin \
  --pipeline_configs "${PIPELINE_CFGS[@]}" "${GT_CFGS[@]}" \
  --overrides "${OVERRIDES[@]}" "${LIDAR_OVERRIDES[@]:-}" \
  --pipeline_overrides "${PIPELINE_OVERRIDES[@]}"
set +x

echo "[lidar_rrt_job] Completed. Outputs: $OUT_DIR"
