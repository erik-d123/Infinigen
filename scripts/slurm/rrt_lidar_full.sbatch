#!/usr/bin/env bash
#SBATCH --job-name=rrt_lidar_full
#SBATCH --output=/scratch/ed1783/infinigen_work/Infinigen/scripts/slurm/logs/rrt_lidar_%j.out
#SBATCH --error=/scratch/ed1783/infinigen_work/Infinigen/scripts/slurm/logs/rrt_lidar_%j.err
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --gres=gpu:l40:1
#SBATCH --account=seas
#SBATCH --partition=all

set -euo pipefail

# Force our own debug log in addition to Slurm's StdOut
DBG_LOG="/scratch/ed1783/infinigen_work/Infinigen/scripts/slurm/logs/rrt_lidar_${SLURM_JOB_ID}.debug.log"
mkdir -p "$(dirname "$DBG_LOG")"
exec > >(tee -a "$DBG_LOG") 2>&1
set -x

# Persistent locations (created by setup script)
PERSIST="/u/ed1783/infinigen_persist"
ENV_PATH="$PERSIST/conda/envs/infinigen"

# Scratch repo and outputs
REPO="/scratch/ed1783/infinigen_work/Infinigen"
OUT_ROOT="/scratch/ed1783/infinigen_runs/${SLURM_JOB_ID}"

# Scene/camera settings
NUM_SCENES=1
FRAME_RANGE_LO=1
FRAME_RANGE_HI=200

mkdir -p "/scratch/ed1783/infinigen_work/Infinigen/scripts/slurm/logs"

echo "[job] JobID=$SLURM_JOB_ID  Host=$(hostname)"
echo "[job] Repo=$REPO"
echo "[job] Out=$OUT_ROOT  Scenes=$NUM_SCENES  Frames=${FRAME_RANGE_LO}-${FRAME_RANGE_HI}"

# Prefer absolute Python from the persistent env to avoid activation issues
PY="$ENV_PATH/bin/python"
if [[ ! -x "$PY" ]]; then
  # Fallback to module + activate
  module load anaconda3 || true
  if ! command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /usr/local/anaconda3/etc/profile.d/conda.sh || true
  fi
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_PATH" || true
    PY="$(command -v python || true)"
  fi
fi
echo "[job] Using Python: ${PY:-not-found}"

# Ensure we run from the repo root so launch_blender finds blender/blender symlink
cd "$REPO"

# Scratch temp
export TMPDIR="${SLURM_TMPDIR:-/scratch/${USER}/tmp}"
mkdir -p "$TMPDIR" "$OUT_ROOT"

"$PY" -c "import sys; print('[env] Python:', sys.version); print('[env] EXE:', sys.executable)"
# Check for bpy in the orchestrator env
"$PY" - <<'PYEOF' || true
try:
    import bpy
    print('[env] bpy OK', bpy.app.version_string)
except Exception as e:
    print('[env] WARNING: bpy import failed in orchestrator env:', repr(e))
PYEOF
blender/blender --version || true

# Run Indoors + RRT + exporter bake (global) + LiDAR (camera-dependent)
"$PY" -m infinigen.datagen.manage_jobs \
  --output_folder "$OUT_ROOT" \
  --num_scenes "$NUM_SCENES" \
  --overwrite \
  --configs singleroom.gin rrt_cam_indoors.gin \
  --pipeline_configs \
      local_256GB.gin \
      indoor_background_configs.gin \
      blender_gt.gin \
      infinigen/datagen/configs/gt_options/lidar_bake_textures.gin \
      infinigen/datagen/configs/gt_options/lidar.gin \
  --overrides \
      compute_base_views.min_candidates_ratio=2 \
      compose_indoors.terrain_enabled=False \
      compose_indoors.restrict_single_supported_roomtype=True \
  --pipeline_overrides \
      get_cmd.driver_script='infinigen_examples.generate_indoors' \
      "iterate_scene_tasks.frame_range=[${FRAME_RANGE_LO},${FRAME_RANGE_HI}]"

echo "[job] Done. Outputs under: $OUT_ROOT"
