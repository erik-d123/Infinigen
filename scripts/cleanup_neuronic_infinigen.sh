#!/usr/bin/env bash
# Cautiously list and (optionally) remove scratch and persistent Infinigen paths.
#
# Usage:
#   bash scripts/cleanup_neuronic_infinigen.sh --persist /n/fs/<group>/<user>/infg_persist --scratch /scratch/$USER/infinigen_work [--yes]

set -euo pipefail

PERSIST=""
SCR=""
YES="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --persist) PERSIST="$2"; shift 2;;
    --scratch) SCR="$2"; shift 2;;
    --yes) YES="true"; shift 1;;
    -h|--help)
      sed -n '1,120p' "$0"; exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

[[ -n "$PERSIST" ]] || { echo "--persist is required" >&2; exit 1; }
[[ -n "$SCR" ]] || { echo "--scratch is required" >&2; exit 1; }

echo "[cleanup] Will operate on:"
echo "  PERSIST: $PERSIST"
echo "  SCR    : $SCR"

echo "[cleanup] Listing candidates to remove:"
echo "  Scratch repo dir: $SCR/Infinigen"
echo "  Scratch outputs  : $SCR/smoke_out (example), other */out paths you created"
echo "  Persistent conda : $PERSIST/conda/envs/infinigen (if you chose this env name)"
echo "  Persistent blender: $PERSIST/blender (binary + ~3.5GB), $PERSIST/blender_vendor (site-packages)"

if [[ "$YES" != "true" ]]; then
  echo "[cleanup] Dry-run only. Re-run with --yes to delete."; exit 0
fi

read -r -p "[cleanup] Really remove scratch repo ($SCR/Infinigen)? [y/N] " ans
if [[ "${ans,,}" == y* ]]; then
  rm -rf "$SCR/Infinigen" || true
  echo "[cleanup] Removed $SCR/Infinigen"
fi

read -r -p "[cleanup] Remove scratch outputs under $SCR (manual glob)? [y/N] " ans
if [[ "${ans,,}" == y* ]]; then
  find "$SCR" -maxdepth 1 -mindepth 1 -type d -name "*out*" -print -exec rm -rf {} + || true
  echo "[cleanup] Removed common output folders in $SCR"
fi

read -r -p "[cleanup] Remove persistent Blender ($PERSIST/blender) and vendor ($PERSIST/blender_vendor)? [y/N] " ans
if [[ "${ans,,}" == y* ]]; then
  rm -rf "$PERSIST/blender" "$PERSIST/blender_vendor" || true
  echo "[cleanup] Removed Blender binaries and vendor packages"
fi

read -r -p "[cleanup] Remove persistent conda env at $PERSIST/conda/envs/infinigen? [y/N] " ans
if [[ "${ans,,}" == y* ]]; then
  if command -v conda >/dev/null 2>&1; then
    conda env remove -p "$PERSIST/conda/envs/infinigen" || true
  fi
  rm -rf "$PERSIST/conda/envs/infinigen" || true
  echo "[cleanup] Removed conda env infinigen"
fi

echo "[cleanup] Done."

