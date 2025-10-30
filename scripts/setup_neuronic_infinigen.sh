#!/usr/bin/env bash
# Setup Infinigen for neuronic/Slurm with persistent conda + Blender, and a scratch checkout.
#
# - Installs a persistent conda env (Python 3.11) with orchestration deps
# - Downloads Blender 4.2.x to a persistent folder (no UI, used headless)
# - Installs Blender-side Python packages into a persistent vendor dir
# - Clones the repo to scratch and symlinks the persistent Blender + vendor
# - Verifies Blender can import infinigen and gin
# - Optional: runs a small indoor + LiDAR smoke test on scratch
#
# Usage examples:
#   bash scripts/setup_neuronic_infinigen.sh \
#     --persist /n/fs/<group>/<user>/infg_persist \
#     --scratch /scratch/$USER/infinigen_work \
#     --repo-ssh git@github.com:erik-d123/Infinigen.git \
#     --blender 4.2.2 \
#     --env-name infinigen \
#     --run-smoke
#
# Notes:
# - This script is idempotent: it skips steps that already exist.
# - You must have write access to the chosen persistent directory.

set -euo pipefail

die() { echo "[setup] ERROR: $*" >&2; exit 1; }
log() { echo "[setup] $*"; }

# Defaults
PERSIST=""
SCR="/scratch/${USER}/infinigen_work"
REPO_SSH="git@github.com:erik-d123/Infinigen.git"
BLV="4.2.2"
ENV_NAME="infinigen"
RUN_SMOKE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --persist) PERSIST="$2"; shift 2;;
    --scratch) SCR="$2"; shift 2;;
    --repo-ssh) REPO_SSH="$2"; shift 2;;
    --blender) BLV="$2"; shift 2;;
    --env-name) ENV_NAME="$2"; shift 2;;
    --run-smoke) RUN_SMOKE="true"; shift 1;;
    -h|--help)
      sed -n '1,80p' "$0"; exit 0;;
    *) die "Unknown arg: $1";;
  esac
done

[[ -n "$PERSIST" ]] || die "--persist <PATH> is required (not scratch)."

mkdir -p "$PERSIST" "$SCR"

# Resolve paths
ENV_PATH="$PERSIST/conda/envs/${ENV_NAME}"
BL_ROOT="$PERSIST/blender/blender-${BLV}-linux-x64"
BL_BIN_PERSIST="$PERSIST/blender/blender"
VENDOR_DIR="$PERSIST/blender_vendor"
REPO_DIR="$SCR/Infinigen"

log "Persist: $PERSIST"
log "Scratch : $SCR"
log "Repo SSH: $REPO_SSH"
log "Blender : $BLV"
log "Env    : $ENV_NAME -> $ENV_PATH"

#
# 1) Load conda and create env (Python 3.11)
#
if command -v module >/dev/null 2>&1; then
  module load anaconda3 || true
fi
if ! command -v conda >/dev/null 2>&1; then
  # Try common profile hook
  if [[ -f /usr/local/anaconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    . /usr/local/anaconda3/etc/profile.d/conda.sh || true
  fi
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" 2>/dev/null || true
else
  die "Conda not found. Try: module load anaconda3"
fi

if [[ ! -d "$ENV_PATH" ]]; then
  log "Creating conda env at: $ENV_PATH"
  conda create -y -p "$ENV_PATH" python=3.11
else
  log "Conda env exists: $ENV_PATH"
fi

log "Activating env: $ENV_PATH"
conda activate "$ENV_PATH"

log "Installing orchestration deps into conda env"
python -m pip install --upgrade pip
python -m pip install "gin-config>=0.5" submitit pandas jinja2 "numpy<2" tqdm psutil || die "pip failed"

#
# 2) Persistent Blender install
#
if [[ ! -x "$BL_BIN_PERSIST" ]]; then
  log "Installing Blender $BLV into $PERSIST/blender"
  mkdir -p "$PERSIST/blender"
  cd "$PERSIST/blender"
  if [[ ! -d "$BL_ROOT" ]]; then
    wget -O blender.tar.xz "https://download.blender.org/release/Blender4.2/blender-${BLV}-linux-x64.tar.xz"
    mkdir -p "$BL_ROOT"
    tar -C "$BL_ROOT" --strip-components=1 -xf blender.tar.xz
    rm -f blender.tar.xz
  fi
  ln -sfn "$BL_ROOT/blender" "$BL_BIN_PERSIST"
else
  log "Blender already present: $BL_BIN_PERSIST"
fi

#
# 3) Blender vendor site-packages (persistent)
#
BL_PYBIN="$BL_ROOT/4.2/python/bin/python3.11"
[[ -x "$BL_PYBIN" ]] || die "Blender Python not found at $BL_PYBIN"

log "Installing Blender-side Python deps into $VENDOR_DIR"
mkdir -p "$VENDOR_DIR"
"$BL_PYBIN" -m ensurepip --upgrade || true
"$BL_PYBIN" -m pip install --upgrade pip || true
"$BL_PYBIN" -m pip install --upgrade \
  --target "$VENDOR_DIR" \
  'numpy==1.26.4' \
  'coacd==1.0.7' \
  'trimesh<3.23.0' \
  'shapely<=2.0.5' \
  networkx 'imageio<2.32.0' tqdm matplotlib opencv-python-headless \
  'gin_config>=0.5.0' || die "Blender pip failed"

# Verify vendor modules importable inside Blender
log "Verifying Blender vendor modules (gin, coacd, trimesh, shapely, imageio, cv2)"
python -m infinigen.launch_blender --background -m infinigen.tools.blendscript_path_append -- \
  --python-expr "import sys,os;print('[verify] vendor_in_path=', any(p.endswith('/.blender_site') or p.endswith('/blender_vendor') for p in sys.path)); import gin;print('[verify] gin',gin.__version__); import coacd,trimesh,shapely,imageio; print('[verify] coacd OK, trimesh', getattr(trimesh,'__version__','?'),' shapely', getattr(shapely,'__version__','?'),' imageio', imageio.__version__); os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'; import cv2; print('[verify] cv2', cv2.__version__)" || die "Blender vendor verification failed"

#
# 4) Clone repo to scratch and symlink Blender + vendor
#
if [[ ! -d "$REPO_DIR/.git" ]]; then
  log "Cloning repo to scratch: $REPO_DIR"
  git clone "$REPO_SSH" "$REPO_DIR"
else
  log "Repo exists; pulling latest"
  git -C "$REPO_DIR" pull --ff-only || true
fi

cd "$REPO_DIR"
mkdir -p infinigen/blender
ln -sfn "$BL_BIN_PERSIST" infinigen/blender/blender
ln -sfn "$VENDOR_DIR" .blender_site

#
# 5) Verify Blender can import infinigen + gin
#
log "Verifying Blender launch via launcher"
python -m infinigen.launch_blender --background -m infinigen.tools.blendscript_path_append -- \
  --python-expr "import sys,gin;print('[verify] Blender Python:',sys.version);print('[verify] gin OK',gin.__version__)" || die "Blender verify failed"

log "Setup complete. Repo: $REPO_DIR"

if [[ "$RUN_SMOKE" == "true" ]]; then
  log "Running small indoor + LiDAR smoke test on scratch"
  OUT="$SCR/smoke_out"; mkdir -p "$OUT" "$OUT/tmp"; export TMPDIR="$OUT/tmp"
  python -m infinigen.launch_blender -m infinigen_examples.generate_indoors -- \
    --seed 0 --task coarse \
    --output_folder "$OUT/coarse" \
    -g singleroom.gin \
    -p compose_indoors.terrain_enabled=False compose_indoors.restrict_single_supported_roomtype=True

  bash scripts/bake_export_textures.sh "$OUT/coarse" "$OUT/export" 1024 || true

  python -m infinigen.launch_blender -m lidar.lidar_generator -- \
    "$OUT/coarse/scene.blend" \
    --output_dir "$OUT/lidar" \
    --frames 1-3 \
    --camera Camera \
    --preset VLP-16 \
    --export-bake-dir "$OUT/export/textures" || die "LiDAR smoke failed"
  log "Smoke test done. Outputs at: $OUT"
fi
