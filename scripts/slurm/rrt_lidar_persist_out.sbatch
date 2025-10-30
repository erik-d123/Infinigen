#!/usr/bin/env bash
#SBATCH --job-name=rrt_lidar_full
#SBATCH --output=/u/ed1783/Infinigen/scripts/slurm/logs/rrt_lidar_%j.out
#SBATCH --error=/u/ed1783/Infinigen/scripts/slurm/logs/rrt_lidar_%j.err
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --gres=gpu:l40:1
#SBATCH --account=seas
#SBATCH --partition=all
#SBATCH --chdir=/u/ed1783/Infinigen

set -euo pipefail

# Extra debug log (in addition to Slurm StdOut)
DBG_LOG="/u/ed1783/Infinigen/scripts/slurm/logs/rrt_lidar_${SLURM_JOB_ID}.debug.log"
mkdir -p "$(dirname "$DBG_LOG")"
exec > >(tee -a "$DBG_LOG") 2>&1
set -x

# Persistent env Python (avoid activation issues)
ENV_PATH="/u/ed1783/infinigen_persist/conda/envs/infinigen"
PY="$ENV_PATH/bin/python"
if [[ ! -x "$PY" ]]; then
  module load anaconda3 || true
  # shellcheck disable=SC1091
  source /usr/local/anaconda3/etc/profile.d/conda.sh || true
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_PATH" || true
    PY="$(command -v python || true)"
  fi
fi

echo "[job] Using Python: ${PY:-not-found}"
"$PY" -c "import sys; print('[env] Python:', sys.version); print('[env] EXE:', sys.executable)"

# Check bpy import in orchestrator env (optional)
"$PY" - <<'PYEOF' || true
try:
    import bpy
    print('[env] bpy OK', getattr(bpy.app, 'version_string', '?'))
except Exception as e:
    print('[env] WARNING: bpy import failed in orchestrator env:', repr(e))
PYEOF

# Blender binary must be at repo_root/blender/blender (symlink to persistent Blender)
blender/blender --version || true

# Outputs on scratch
OUT_ROOT="/scratch/ed1783/infinigen_runs/${SLURM_JOB_ID}"
export TMPDIR="${SLURM_TMPDIR:-/scratch/${USER}/tmp}"
mkdir -p "$TMPDIR" "$OUT_ROOT"

echo "[job] OUT_ROOT=$OUT_ROOT  TMPDIR=$TMPDIR"

# Scene/camera settings
NUM_SCENES=1
FRAME_RANGE_LO=1
FRAME_RANGE_HI=200

# Indoors + RRT + exporter bake + LiDAR
"$PY" -m infinigen.datagen.manage_jobs \
  --output_folder "$OUT_ROOT" \
  --num_scenes "$NUM_SCENES" \
  --overwrite \
  --configs singleroom.gin rrt_cam_indoors.gin \
  --pipeline_configs \
      local_256GB.gin \
      infinigen/datagen/configs/data_schema/monocular_video.gin \
      indoor_background_configs.gin \
      blender_gt.gin \
      infinigen/datagen/configs/gt_options/lidar_bake_textures.gin \
      infinigen/datagen/configs/gt_options/lidar.gin \
  --overrides \
      compute_base_views.min_candidates_ratio=2 \
      compose_indoors.terrain_enabled=False \
      compose_indoors.restrict_single_supported_roomtype=True \
      lidar.lidar_config.LidarConfig.ply_binary=True \
  --pipeline_overrides \
      get_cmd.driver_script='infinigen_examples.generate_indoors' \
      "iterate_scene_tasks.frame_range=[${FRAME_RANGE_LO},${FRAME_RANGE_HI}]" \
      queue_export_bake_textures.submit_cmd=@local_submit_cmd \
      queue_lidar.submit_cmd=@local_submit_cmd \
      queue_export_bake_textures.image_res=1024

echo "[job] Done. Outputs under: $OUT_ROOT"
