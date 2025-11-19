#!/usr/bin/env bash
# Submit Indoors (coarse) + RRT camera + LiDAR ground truth via Slurm,
# writing outputs to scratch by default.
#
# Usage examples (run on login node):
#   scripts/slurm/run_indoor_lidar_scratch.sh --frames 1-100 --scenes 1 \
#     --account seas --partition all --name my_lidar_rrt
#
# Options:
#   --frames   Frame range, e.g. 1-100 (default: 1-48)
#   --scenes   Number of scenes (default: 1)
#   --seed     Specific seed (hex/string ok) (default: unset)
#   --account  Slurm account (exported to INFINIGEN_SLURMPARTITION)
#   --partition Slurm partition name (adds gin override)
#   --out      Output root dir (default: /scratch/$USER/infinigen_runs/<name or timestamp>)
#   --name     Name suffix used in default --out
#   --repo     Path to repo (default: current working dir)

set -euo pipefail

FRAMES="1-48"
SCENES=1
SEED=""
ACCOUNT=""
PARTITION=""
OUT_ROOT=""
NAME="lidar_rrt"
REPO_DIR="$(pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --frames) FRAMES="$2"; shift 2;;
    --scenes) SCENES="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --account) ACCOUNT="$2"; shift 2;;
    --partition) PARTITION="$2"; shift 2;;
    --out) OUT_ROOT="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --repo) REPO_DIR="$2"; shift 2;;
    -h|--help) sed -n '1,120p' "$0"; exit 0;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

if ! command -v sbatch >/dev/null 2>&1; then
  echo "[run_indoor_lidar_scratch] ERROR: sbatch not found. Run on login node (e.g., neuronic) and module load slurm if needed." >&2
  exit 127
fi

# Default OUT_ROOT to scratch
if [[ -z "$OUT_ROOT" ]]; then
  TS="$(date +"%Y%m%d_%H%M%S")"
  OUT_ROOT="/scratch/${USER}/infinigen_runs/${NAME}_${TS}"
fi

mkdir -p "$OUT_ROOT"

echo "[run_indoor_lidar_scratch] Repo:    $REPO_DIR"
echo "[run_indoor_lidar_scratch] Out:     $OUT_ROOT"
echo "[run_indoor_lidar_scratch] Scenes:  $SCENES  Frames: $FRAMES  Seed: ${SEED:-<random>}"

# Account env for slurm.gin (uses ENVVAR_INFINIGEN_SLURMPARTITION as account string)
if [[ -n "$ACCOUNT" ]]; then
  export INFINIGEN_SLURMPARTITION="$ACCOUNT"
  echo "[run_indoor_lidar_scratch] Using account: $INFINIGEN_SLURMPARTITION"
fi

# Build gin pipeline overrides
LO="${FRAMES%-*}"
HI="${FRAMES#*-}"
PIPE_OVR=(
  "get_cmd.driver_script='infinigen_examples.generate_indoors'"
  "iterate_scene_tasks.frame_range=[${LO},${HI}]"
)
if [[ -n "$PARTITION" ]]; then
  PIPE_OVR+=("slurm_submit_cmd.slurm_partition='${PARTITION}'")
fi

CMD=(
  python -m infinigen.datagen.manage_jobs
  --output_folder "$OUT_ROOT"
  --num_scenes "$SCENES"
  --pipeline_configs compute_platform/slurm.gin data_schema/monocular.gin gt_options/lidar_gt.gin
  --pipeline_overrides "${PIPE_OVR[@]}"
  --configs singleroom.gin rrt_cam_indoors.gin
  -p compose_indoors.terrain_enabled=False
)

if [[ -n "$SEED" ]]; then
  CMD+=(--specific_seed "$SEED")
fi

(
  cd "$REPO_DIR"
  echo "[run_indoor_lidar_scratch] Submitting pipeline via Slurm (manage_jobs will dispatch tasks)..."
  "${CMD[@]}"
)

echo "[run_indoor_lidar_scratch] Submitted. Outputs under: $OUT_ROOT"

