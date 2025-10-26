#!/usr/bin/env python3

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _filter_blender_cli(argv: list[str]) -> list[str]:
    """Remove Blender launch flags and return only pytest arguments."""
    if "--" in argv:
        idx = argv.index("--")
        return argv[idx + 1 :]

    filtered: list[str] = []
    skip_next = False
    blender_prefixes = {
        "-b",  # shorthand for --background
        "--background",
        "--factory-startup",
        "-noaudio",
    }
    blender_options_with_value = {
        "--python",
        "--python-text",
        "--python-expr",
    }

    for arg in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in blender_prefixes:
            continue
        if arg in blender_options_with_value:
            skip_next = True
            continue
        filtered.append(arg)

    return filtered


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    pytest_args = _filter_blender_cli(list(argv))
    pytest_args = [arg for arg in pytest_args if arg != "--factory-startup"]
    if not pytest_args:
        pytest_args = ["tests/lidar"]
    if "-q" not in pytest_args and "--quiet" not in pytest_args:
        pytest_args = ["-q", *pytest_args]
    import pytest
    return pytest.main(pytest_args)


if __name__ == "__main__":
    sys.exit(main())
