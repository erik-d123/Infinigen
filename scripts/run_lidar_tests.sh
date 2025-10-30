#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Use module mode so launch_blender injects .blender_site via blendscript_path_append,
# then run the pytest driver script. Do NOT put a "--" before the --python.
python -m infinigen.launch_blender -m infinigen.tools.blendscript_path_append \
  --python "${ROOT}/scripts/run_lidar_pytest.py" -- --factory-startup "$@"
