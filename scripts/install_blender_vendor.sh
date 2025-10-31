#!/usr/bin/env bash
# Install Python packages into Blender's embedded Python and place them in a
# repo-local vendor directory (default: .blender_site). This makes packages
# importable when running Blender via infinigen.launch_blender.
#
# Usage examples (from the repo root):
#   bash scripts/install_blender_vendor.sh
#   bash scripts/install_blender_vendor.sh --vendor-dir /u/$USER/Infinigen/.blender_site
#   bash scripts/install_blender_vendor.sh --packages "scikit-image==0.21.0"  # extra packages
#
# The script will:
#   - Locate Blender's embedded python using infinigen.launch_blender
#   - Create the vendor dir if missing
#   - Install a curated set of wheels into the vendor dir (versions pinned for compatibility)
#   - Verify selected imports inside Blender with the vendor path injected

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/.blender_site"
EXTRA_PKGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vendor-dir) VENDOR_DIR="$2"; shift 2;;
    --packages) EXTRA_PKGS="$2"; shift 2;;
    -h|--help)
      sed -n '1,80p' "$0"; exit 0;;
    *) echo "[install_blender_vendor] Unknown option: $1" >&2; exit 1;;
  esac
done

echo "[install_blender_vendor] Repo:        $REPO_ROOT"
echo "[install_blender_vendor] Vendor dir:  $VENDOR_DIR"

# 1) Locate Blender's embedded Python via launcher
echo "[install_blender_vendor] Locating Blender's Python via infinigen.launch_blender..."
BL_PY="$(python -m infinigen.launch_blender --background --python-expr 'import sys; print(sys.executable)' 2>/dev/null | sed -n '/^\//p' | head -n1 || true)"
if [[ -z "${BL_PY:-}" || ! -x "$BL_PY" ]]; then
  echo "[install_blender_vendor] ERROR: Failed to locate Blender's embedded Python." >&2
  echo "Ensure Blender is set up under repo 'blender/blender' or system Blender is configured with launch_blender." >&2
  exit 1
fi
echo "[install_blender_vendor] Blender Python: $BL_PY"

# 2) Prepare vendor dir and upgrade pip
mkdir -p "$VENDOR_DIR"
echo "[install_blender_vendor] Ensuring pip in Blender's Python..."
"$BL_PY" -m ensurepip --upgrade || true
"$BL_PY" -m pip install --upgrade pip || true

# 3) Install curated package set compatible with Infinigen/Exporter and Indoors
echo "[install_blender_vendor] Installing packages into vendor: $VENDOR_DIR"
"$BL_PY" -m pip install --upgrade --target "$VENDOR_DIR" \
  'numpy==1.26.4' \
  'scipy==1.11.4' \
  'scikit-learn==1.4.2' \
  'scikit-image==0.21.0' \
  'lazy_loader>=0.3' \
  'tifffile>=2022.8.12' \
  'PyWavelets>=1.4.1' \
  'packaging>=21' \
  psutil pillow matplotlib \
  'imageio<2.32.0' \
  opencv-python-headless \
  'shapely<=2.0.5' \
  'trimesh<3.23.0' \
  networkx \
  'coacd==1.0.7' \
  tqdm \
  'gin_config>=0.5.0' \
  joblib threadpoolctl ${EXTRA_PKGS}

# 4) Verify imports inside Blender with .blender_site injected
echo "[install_blender_vendor] Verifying imports inside Blender..."
python -m infinigen.launch_blender --background -m infinigen.tools.blendscript_path_append -- \
  --python-expr "import sys; print('[verify] vendor_in_path=', any(p.endswith('/.blender_site') for p in sys.path)); \
import numpy, scipy, sklearn, psutil, imageio, shapely, trimesh; \
print('[verify] numpy', numpy.__version__, 'scipy', scipy.__version__, 'sklearn', sklearn.__version__)" || {
  echo "[install_blender_vendor] WARNING: Verification failed. Ensure .blender_site is added to sys.path at Blender start." >&2
}

echo "[install_blender_vendor] Done."
