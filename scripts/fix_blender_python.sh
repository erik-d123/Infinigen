#!/usr/bin/env bash
# Install Python deps into the Blender embedded interpreter used by Infinigen,
# and make them visible via a repo-local vendor directory (.blender_site).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/.blender_site"

echo "[fix_blender_python] Repo: $REPO_ROOT"

echo "[fix_blender_python] Locating Blender's embedded Python via infinigen.launch_blender..."
PYBIN="$(python -m infinigen.launch_blender --background --python-expr 'import sys; print(sys.executable)' 2>/dev/null | sed -n '/^\//p' | head -n1)"
if [[ -z "${PYBIN:-}" || ! -x "$PYBIN" ]]; then
  echo "[fix_blender_python] Could not detect Blender's Python automatically."
  echo "[fix_blender_python] Trying bundle path heuristic..."
  PYBIN="$(ls -1 "$REPO_ROOT"/Blender.app/Contents/Resources/*/python/bin/python3* 2>/dev/null | head -n1 || true)"
fi

if [[ -z "${PYBIN:-}" || ! -x "$PYBIN" ]]; then
  echo "[fix_blender_python] ERROR: Failed to locate Blender's embedded Python."
  echo "Please ensure you installed Infinigen as a Blender-Python script and that Blender.app exists under the repo."
  exit 1
fi

echo "[fix_blender_python] Using Blender Python: $PYBIN"
mkdir -p "$VENDOR_DIR"

echo "[fix_blender_python] Upgrading pip in Blender's Python (user space if needed)..."
"$PYBIN" -m ensurepip --upgrade || true
"$PYBIN" -m pip install --upgrade pip || true

echo "[fix_blender_python] Installing packages into vendor dir: $VENDOR_DIR"
# Pin to versions compatible with Infinigen: numpy<2, shapely<=2.0.5, trimesh<3.23.0
"$PYBIN" -m pip install --upgrade \
  --target "$VENDOR_DIR" \
  'numpy<2' 'shapely<=2.0.5' 'trimesh<3.23.0' networkx 'imageio<2.32.0' || {
    echo "[fix_blender_python] WARNING: pip install reported a non-zero exit code; continuing."
  }

echo "[fix_blender_python] Verifying import inside Blender..."
python -m infinigen.launch_blender --background -m infinigen.tools.blendscript_path_append -- \
  --python-expr "import sys; print('[verify] vendor_in_path=', any(p.endswith('/.blender_site') for p in sys.path)); import trimesh; print('[verify] trimesh OK')" || {
  echo "[fix_blender_python] WARNING: Verification failed. Ensure .blender_site is added to sys.path at Blender start."
}

echo "[fix_blender_python] Done."
