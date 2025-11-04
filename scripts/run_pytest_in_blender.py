#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run pytest inside Blender's Python environment via infinigen.launch_blender."""

from __future__ import annotations

import os
import sys


def _extend_sys_path_from_env():
    """Append any site-packages passed in via environment variables."""
    env_keys = (
        "INFINIGEN_SITE_PACKAGES",
        "PYTHONPATH",  # fallback
    )
    for key in env_keys:
        paths = os.environ.get(key)
        if not paths:
            continue
        for path in paths.split(os.pathsep):
            if path and path not in sys.path:
                sys.path.append(path)


_extend_sys_path_from_env()

try:
    import pytest
except (
    ImportError
) as exc:  # pragma: no cover - blender python should have pytest from dev install
    raise RuntimeError("pytest is required to run Blender-based tests") from exc


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv

    if "--" in argv:
        idx = argv.index("--")
        pytest_args = argv[idx + 1 :]
    else:
        pytest_args = argv[1:]

    if not pytest_args:
        pytest_args = ["tests"]

    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
