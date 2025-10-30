#!/usr/bin/env bash
# Install coacd (and a compatible numpy) into Blender's embedded Python
# so that Infinigen's exporter can run under infinigen.launch_blender.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[install_coacd] Repo: $REPO_ROOT"

# Locate Blender's embedded Python via the launcher
echo "[install_coacd] Locating Blender's Python..."
PYBIN="$(python -m infinigen.launch_blender --background --python-expr 'import sys; print(sys.executable)' 2>/dev/null | grep -E '^/' | head -n1 || true)"
if [[ -z "${PYBIN:-}" || ! -x "$PYBIN" ]]; then
  echo "[install_coacd] Auto-detect failed, trying bundle heuristic..."
  PYBIN="$(ls -1 "$REPO_ROOT"/Blender.app/Contents/Resources/*/python/bin/python3* 2>/dev/null | head -n1 || true)"
fi
if [[ -z "${PYBIN:-}" || ! -x "$PYBIN" ]]; then
  echo "[install_coacd] ERROR: Could not locate Blender's embedded Python."
  echo "Ensure Infinigen is installed as a Blender-Python script and Blender.app exists under the repo."
  exit 1
fi
echo "[install_coacd] Using Blender Python: $PYBIN"

# Repo-local vendor directory that Blender loads (blendscript_path_append.py adds it to sys.path)
VENDOR_DIR="$REPO_ROOT/.blender_site"
mkdir -p "$VENDOR_DIR"

echo "[install_coacd] Ensuring pip in Blender's Python..."
"$PYBIN" -m ensurepip --upgrade || true
"$PYBIN" -m pip install --upgrade pip || true

echo "[install_coacd] Installing dependencies into $VENDOR_DIR ..."
# Pins compatible with Blender 3.11 and Infinigen
"$PYBIN" -m pip install --upgrade \
  --target "$VENDOR_DIR" \
  'numpy==1.26.4' \
  'coacd==1.0.7' \
  'trimesh<3.23.0' \
  'shapely<=2.0.5' \
  networkx 'imageio<2.32.0' tqdm matplotlib opencv-python-headless

echo "[install_coacd] Verifying inside Blender..."
# Ensure .blender_site is in sys.path (blendscript_path_append.py does this), then import coacd
python -m infinigen.launch_blender --background -m infinigen.tools.blendscript_path_append -- \
  --python-expr "import os, sys; print('[verify] vendor_in_path=', any(p.endswith('/.blender_site') for p in sys.path)); import coacd; print('[verify] coacd OK'); import trimesh, shapely, networkx, imageio; print('[verify] trimesh/shapely/networkx/imageio', imageio.__version__); os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'; import cv2; print('[verify] cv2 OK', cv2.__version__)" || {
  echo "[install_coacd] WARNING: Verification failed. Check that .blender_site is added to sys.path at Blender start."
}

echo "[install_coacd] Done."
