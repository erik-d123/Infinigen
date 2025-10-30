#!/usr/bin/env bash
# Submit an indoor RRT video + LiDAR pipeline job via Slurm.
#
# Usage:
#   scripts/slurm/submit_lidar_rrt.sh <job_name> <gpu_type> <num_gpus> \
#     [--scenes N] [--frames 1-200] [--seed 0] \
#     [--out outputs/video_dynamic_indoor] \
#     [--time 01-00:00:00] [--account ACC] [--partition PART] \
#     [--env infinigen] [--ocmesh true|false] [--no-bake-normals]
#
# gpu_type: one of 1080, 2080, 3090, 6000, a40, a6000

set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 <job_name> <gpu_type> <num_gpus> [options]" >&2
  exit 1
fi

JOB_NAME="$1"; shift
GPU_TYPE="$1"; shift
NUM_GPUS="$1"; shift

# Defaults
NUM_SCENES=${NUM_SCENES:-1}
FRAME_RANGE=${FRAME_RANGE:-"1-200"}
SEED=${SEED:-0}
OUT_DIR=${OUT_DIR:-"outputs/video_dynamic_indoor"}
TIME_LIMIT=${TIME_LIMIT:-"01-00:00:00"}
SLURM_ACCOUNT_OPT=""
SLURM_PARTITION_OPT=""
CONDA_ENV=${INFINIGEN_CONDA_ENV:-"infinigen"}
ENABLE_OCMESH=${ENABLE_OCMESH:-"false"}
NO_BAKE_NORMALS=${NO_BAKE_NORMALS:-"false"}

# Ensure sbatch is available (must run on cluster login node)
if ! command -v sbatch >/dev/null 2>&1; then
  echo "[submit_lidar_rrt] ERROR: 'sbatch' not found in PATH." >&2
  echo "Run this on your cluster login node (e.g., 'ssh neuronic') and, if needed, 'module load slurm'." >&2
  exit 127
fi

# Parse optional flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenes) NUM_SCENES="$2"; shift 2;;
    --frames) FRAME_RANGE="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --out) OUT_DIR="$2"; shift 2;;
    --time) TIME_LIMIT="$2"; shift 2;;
    --account) SLURM_ACCOUNT_OPT="--account=$2"; shift 2;;
    --partition) SLURM_PARTITION_OPT="--partition=$2"; shift 2;;
    --env) CONDA_ENV="$2"; shift 2;;
    --ocmesh) ENABLE_OCMESH="$2"; shift 2;;
    --no-bake-normals) NO_BAKE_NORMALS="true"; shift 1;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

# GPU resource presets (per GPU)
case "$GPU_TYPE" in
  1080) GPU_STRING="gtx_1080"; CPUS_PER_GPU=2; MEM_PER_GPU=25600; VRAM=11;;
  2080) GPU_STRING="rtx_2080"; CPUS_PER_GPU=4; MEM_PER_GPU=25600; VRAM=11;;
  6000) GPU_STRING="rtx_6000"; CPUS_PER_GPU=4; MEM_PER_GPU=25600; VRAM=24;;
  3090) GPU_STRING="rtx_3090"; CPUS_PER_GPU=9; MEM_PER_GPU=51200; VRAM=24;;
  a40)  GPU_STRING="a40";      CPUS_PER_GPU=9; MEM_PER_GPU=51200; VRAM=48;;
  a6000)GPU_STRING="a6000";    CPUS_PER_GPU=9; MEM_PER_GPU=51200; VRAM=48;;
  l40)  GPU_STRING="l40";      CPUS_PER_GPU=9; MEM_PER_GPU=51200; VRAM=48;;
  *) echo "invalid GPU type: $GPU_TYPE" >&2; exit 1;;
esac

TOTAL_CPUS=$((CPUS_PER_GPU * NUM_GPUS))
TOTAL_MEM=$((MEM_PER_GPU * NUM_GPUS))

# Best-effort account auto-detect if not provided; default to 'seas'
if [ -z "$SLURM_ACCOUNT_OPT" ]; then
  if command -v sacctmgr >/dev/null 2>&1; then
    ACC=$(sacctmgr -n -P show assoc where user=$USER format=account 2>/dev/null | head -n1 || true)
    if [ -n "$ACC" ]; then SLURM_ACCOUNT_OPT="--account=$ACC"; else SLURM_ACCOUNT_OPT="--account=seas"; fi
  else
    SLURM_ACCOUNT_OPT="--account=seas"
  fi
fi

# Default partition to 'all' if not provided
if [ -z "$SLURM_PARTITION_OPT" ]; then
  SLURM_PARTITION_OPT="--partition=all"
fi

echo "Submitting $JOB_NAME: $NUM_SCENES scenes, frames $FRAME_RANGE, seed $SEED"
echo "GPU: $NUM_GPUS x $GPU_STRING; CPUs: $TOTAL_CPUS; Mem: $((TOTAL_MEM/1000))GB; Time: $TIME_LIMIT"
echo "Account: ${SLURM_ACCOUNT_OPT#--account=}  Partition: ${SLURM_PARTITION_OPT#--partition=}  Env: $CONDA_ENV"

CURR_DIR=$(pwd)

sbatch \
  --job-name="$JOB_NAME" \
  --output="scripts/slurm/${JOB_NAME}.out" \
  --time="$TIME_LIMIT" \
  --cpus-per-task="$TOTAL_CPUS" \
  --mem="$TOTAL_MEM" \
  --gres="gpu:${GPU_STRING}:${NUM_GPUS}" \
  --chdir="$CURR_DIR" \
  $SLURM_ACCOUNT_OPT $SLURM_PARTITION_OPT \
  --export=ALL,\
JOB_OUTPUT_FOLDER="$OUT_DIR",\
JOB_NUM_SCENES="$NUM_SCENES",\
JOB_FRAME_RANGE="$FRAME_RANGE",\
JOB_SEED="$SEED",\
JOB_ENABLE_OCMESH="$ENABLE_OCMESH",\
JOB_NO_BAKE_NORMALS="$NO_BAKE_NORMALS",\
JOB_CONDA_ENV="$CONDA_ENV" \
  scripts/slurm/lidar_rrt_job.sh
