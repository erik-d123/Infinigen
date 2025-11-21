#!/usr/bin/env bash
#SBATCH --job-name=rrt_lidar_direct
#SBATCH --output=/u/ed1783/Infinigen/scripts/slurm/logs/rrt_lidar_direct_%j.out
#SBATCH --error=/u/ed1783/Infinigen/scripts/slurm/logs/rrt_lidar_direct_%j.err
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
DBG_LOG="/u/ed1783/Infinigen/scripts/slurm/logs/rrt_lidar_direct_${SLURM_JOB_ID}.debug.log"
mkdir -p "$(dirname "$DBG_LOG")"
exec > >(tee -a "$DBG_LOG") 2>&1
set -x

ENV_PATH="/u/ed1783/infinigen_persist/conda/envs/infinigen"
PY="$ENV_PATH/bin/python"
BLENDER_BIN="blender/blender"

# Make conda site-packages visible to Blender (easiest way to reuse your env)
BLENDER_CONDA_SITE="$ENV_PATH/lib/python3.11/site-packages"
export PYTHONPATH="$BLENDER_CONDA_SITE:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
echo "[job] BLENDER_CONDA_SITE=$BLENDER_CONDA_SITE"

OUT_ROOT="/scratch/ed1783/infinigen_runs/${SLURM_JOB_ID}"
TMPDIR_DEFAULT="/scratch/${USER}/tmp"
export TMPDIR="${SLURM_TMPDIR:-$TMPDIR_DEFAULT}"
mkdir -p "$TMPDIR" "$OUT_ROOT"

echo "[job] Using Python: $PY"
"$PY" -c "import sys; print('[env] Python:', sys.version); print('[env] EXE:', sys.executable)"
"$PY" - <<'PYEOF' || true
try:
    import bpy
    print('[env] bpy OK', getattr(bpy.app, 'version_string', '?'))
except Exception as e:
    print('[env] WARNING: bpy import failed in orchestrator env:', repr(e))
PYEOF
"$BLENDER_BIN" --version || true

echo "[job] OUT_ROOT=$OUT_ROOT  TMPDIR=$TMPDIR"

# 1) Generate Indoors coarse scene with RRT (single seed)
# Run the Indoors driver as a module to keep package-relative imports valid; launcher adds --background
"$PY" -m infinigen.launch_blender -m infinigen.tools.blendscript_path_append \
  --python-expr "import sys, runpy; \
sys.argv=['Blender','--seed','0','--task','coarse','--output_folder','${OUT_ROOT}/coarse','-g','singleroom.gin','-p','compose_indoors.terrain_enabled=False','compose_indoors.restrict_single_supported_roomtype=True','-d']; \
runpy.run_module('infinigen_examples.generate_indoors', run_name='__main__')"

# Guard: Indoors generation must create a scene.blend. If not, abort early so we don't run bake on a missing folder.
if [[ ! -f "$OUT_ROOT/coarse/scene.blend" ]]; then
  echo "[job] ERROR: Indoors coarse generation failed (missing $OUT_ROOT/coarse/scene.blend). Check Blender output above."
  exit 2
fi

# 2) Generate LiDAR directly from Principled materials
"$PY" -m infinigen.launch_blender -m infinigen.lidar.lidar_generator -- \
  "$OUT_ROOT/coarse/scene.blend" \
  --output_dir "$OUT_ROOT/lidar" \
  --frames 1-200 \
  --camera Camera \
  --preset VLP-16 \
  --ply-binary

echo "[job] Done. Outputs under: $OUT_ROOT"
