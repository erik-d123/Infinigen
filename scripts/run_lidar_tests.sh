#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m infinigen.launch_blender -s "${ROOT}/scripts/run_lidar_pytest.py" -- --factory-startup "$@"
