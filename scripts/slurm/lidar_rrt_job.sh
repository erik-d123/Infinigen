#!/usr/bin/env bash
# Slurm job body for indoor RRT video + LiDAR pipeline
set -euo pipefail

echo "[lidar_rrt_job] Host: $(hostname)  User: $USER"
echo "[lidar_rrt_job] Output: ${JOB_OUTPUT_FOLDER:-unset}  Scenes: ${JOB_NUM_SCENES:-unset}  Frames: ${JOB_FRAME_RANGE:-unset}  Seed: ${JOB_SEED:-unset}"

# Some clusters' /etc/bashrc references variables that are unset under `set -u`.
# Safely source user shell init without strict -u/-e so environment (conda/mamba) is available.
if [ -f "$HOME/.bashrc" ]; then
  set +u
  set +e
  export BASHRCSOURCED=1
  source "$HOME/.bashrc"
  set -e
  set -u
fi

# Activate environment (conda or mamba)
# Try to make `conda activate` available even if ~/.bashrc didn't set it up
if ! command -v conda >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load "${JOB_CONDA_MODULE:-anaconda3}" || true
  fi
fi
if command -v conda >/dev/null 2>&1; then
  # Initialize conda shell function if needed
  eval "$(conda shell.bash hook)" 2>/dev/null || {
    base_dir="$(conda info --base 2>/dev/null || echo '')"
    if [ -n "$base_dir" ] && [ -f "$base_dir/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1090
      . "$base_dir/etc/profile.d/conda.sh"
    fi
  }
fi

if command -v conda >/dev/null 2>&1; then
  if [[ -n "${JOB_CONDA_ENV_PATH:-}" && -d "${JOB_CONDA_ENV_PATH:-}" ]]; then
    conda activate "${JOB_CONDA_ENV_PATH}" || true
  elif conda env list | grep -q "${JOB_CONDA_ENV:-infinigen}"; then
    conda activate "${JOB_CONDA_ENV:-infinigen}" || true
  fi
elif command -v mamba >/dev/null 2>&1; then
  if [[ -n "${JOB_CONDA_ENV_PATH:-}" && -d "${JOB_CONDA_ENV_PATH:-}" ]]; then
    mamba activate "${JOB_CONDA_ENV_PATH}" || true
  else
    mamba activate "${JOB_CONDA_ENV:-infinigen}" || true
  fi
else
  echo "[lidar_rrt_job] WARNING: could not activate conda env ${JOB_CONDA_ENV:-infinigen}" >&2
fi
echo "[lidar_rrt_job] Conda env: ${CONDA_DEFAULT_ENV:-<none>}"

# Try environment modules for Python 3.10+ if conda was unavailable
if ! command -v python >/dev/null 2>&1 || ! python -c 'import sys; exit(0 if sys.version_info[:2] >= (3,10) else 1)'; then
  if [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    . /etc/profile.d/modules.sh || true
  fi
  if command -v module >/dev/null 2>&1; then
    # Allow override of module name, default to a common python/3.10 module
    PY_MOD="${JOB_PYTHON_MODULE:-python/3.10}"
    module load "$PY_MOD" || true
  fi
fi

# Select python binary (env override allowed)
PY_BIN="${JOB_PYTHON_BIN:-python}"
echo "[lidar_rrt_job] Using Python: $(command -v "$PY_BIN" || echo not-found)"
"$PY_BIN" --version || true
# Scratch tmp directory
export TMPDIR="${SLURM_TMPDIR:-/scratch/${USER}/tmp}"
mkdir -p "$TMPDIR"
# Ensure exporter deps are visible to Blender if needed
if [ -d ".blender_site" ]; then
  export PYTHONPATH="$(pwd)/.blender_site:${PYTHONPATH:-}"
fi

OUT_DIR="${JOB_OUTPUT_FOLDER:-outputs/video_dynamic_indoor}"
SCENES="${JOB_NUM_SCENES:-1}"
FRAMES="${JOB_FRAME_RANGE:-1-200}"
SEED="${JOB_SEED:-0}"
OCMESH="${JOB_ENABLE_OCMESH:-false}"
NO_BAKE_N="false"

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

set -x
"$PY_BIN" -m infinigen.datagen.manage_jobs \
  --output_folder "$OUT_DIR" \
  --num_scenes "$SCENES" \
  --specific_seed "$SEED" \
  --configs singleroom.gin rrt_cam_indoors.gin \
  --pipeline_configs "${PIPELINE_CFGS[@]}" "${GT_CFGS[@]}" \
  --overrides "${OVERRIDES[@]}" "${LIDAR_OVERRIDES[@]:-}" \
  --pipeline_overrides "${PIPELINE_OVERRIDES[@]}"
set +x

echo "[lidar_rrt_job] Completed. Outputs: $OUT_DIR"
