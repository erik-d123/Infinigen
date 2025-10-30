#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np

from lidar.lidar_io import save_ply


def test_viewer_read_ply_ascii(tmp_path: Path):
    # Create a tiny PLY via our own writer (ASCII by default)
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    data = {
        "points": pts,
        "intensity": np.array([128], dtype=np.uint8),
        "ring": np.array([0], dtype=np.uint16),
        "azimuth": np.array([0.0], dtype=np.float32),
        "elevation": np.array([0.0], dtype=np.float32),
        "return_id": np.array([1], dtype=np.uint8),
        "num_returns": np.array([1], dtype=np.uint8),
        "reflectivity": np.array([0.5], dtype=np.float32),
    }
    path = tmp_path / "tiny.ply"
    save_ply(path, data, binary=False)
    # Simple header/content sanity (no viewer deps required)
    text = path.read_text().splitlines()
    assert text[0].strip() == "ply"
    assert any("property uchar intensity" in ln for ln in text)
    # Verify one vertex line present after header
    end_hdr = text.index("end_header")
    assert len(text) == end_hdr + 1 + 1
