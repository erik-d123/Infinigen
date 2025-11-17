# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.
# PLY writer and frame‑transform helpers for indoor LiDAR

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import bpy  # only needed by world_to_frame_matrix
except Exception:
    bpy = None

# single source of truth for camera<-sensor rotation
from lidar.lidar_scene import sensor_to_camera_rotation


def world_to_frame_matrix(camera_obj, frame: str = "sensor") -> np.ndarray:
    """Return 4x4 transform world→{world|camera|sensor} for PLY export.

    Sensor frame is defined as +X forward, +Y left, +Z up. Blender camera uses
    +X right, +Y up, -Z forward.
    """
    if frame == "world":
        return np.eye(4, dtype=float)

    # world -> camera
    R_wc = np.array(camera_obj.matrix_world.to_3x3(), dtype=float)
    t_wc = np.array(camera_obj.matrix_world.translation, dtype=float)
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    Twc_inv = np.eye(4, dtype=float)
    Twc_inv[:3, :3] = R_cw
    Twc_inv[:3, 3] = t_cw

    if frame == "camera":
        return Twc_inv

    # camera <- sensor rotation (R_cs)
    R_cs = np.array(sensor_to_camera_rotation(), dtype=float)
    # world -> sensor = (camera -> sensor) @ (world -> camera)
    R_sc = R_cs.T
    Tcw = Twc_inv
    Tsw = np.eye(4, dtype=float)
    Tsw[:3, :3] = R_sc @ Tcw[:3, :3]
    Tsw[:3, 3] = R_sc @ Tcw[:3, 3]
    return Tsw


# Fixed base order; append optional fields if present.
_BASE_LAYOUT = [
    ("x", "f4"),
    ("y", "f4"),
    ("z", "f4"),
    ("intensity", "u1"),
    ("ring", "u2"),
    ("azimuth", "f4"),
    ("elevation", "f4"),
    ("return_id", "u1"),
    ("num_returns", "u1"),
]
_OPT_FIELDS = [
    ("range_m", "f4"),
    ("cos_incidence", "f4"),
    ("mat_class", "u1"),
    ("reflectivity", "f4"),
    ("transmittance", "f4"),
    # normals written if provided as ("normals", Nx3) or ("nx","ny","nz")
]


def _coerce_col(data: Dict, key: str, dtype: str, N: int) -> Optional[np.ndarray]:
    """Fetch and coerce a 1D column from `data` if present and sized for N."""
    if key not in data:
        return None
    arr = np.asarray(data[key])
    if arr.ndim != 1 or arr.shape[0] != N:
        raise ValueError(f"{key}: expected shape ({N},), got {arr.shape}")
    return arr.astype(dtype, copy=False)


def _coerce_points(pts) -> np.ndarray:
    """Validate and coerce a (N, 3) points array."""
    P = np.asarray(pts)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {P.shape}")
    return P.astype("f4", copy=False)


def _detect_normals(data: Dict, N: int) -> Optional[np.ndarray]:
    """Detect normals as an (N, 3) array, supporting both packed and split forms."""
    if "normals" in data:
        n = np.asarray(data["normals"])
        if n.ndim != 2 or n.shape != (N, 3):
            raise ValueError(f"normals must be (N,3), got {n.shape}")
        return n.astype("f4", copy=False)
    # legacy triplets
    have = all(k in data for k in ("nx", "ny", "nz"))
    if have:
        nx = np.asarray(data["nx"]).astype("f4", copy=False)
        ny = np.asarray(data["ny"]).astype("f4", copy=False)
        nz = np.asarray(data["nz"]).astype("f4", copy=False)
        for a in (nx, ny, nz):
            if a.ndim != 1 or a.shape[0] != N:
                raise ValueError("nx/ny/nz must be (N,)")
        return np.stack([nx, ny, nz], axis=1)
    return None


def _build_header(
    N: int, have: Dict[str, bool], have_normals: bool, binary: bool
) -> str:
    """Build a PLY header string for the present columns and format."""
    fmt = "binary_little_endian 1.0" if binary else "ascii 1.0"
    lines = [
        "ply",
        f"format {fmt}",
        f"element vertex {N}",
    ]
    # base props
    lines += [
        "property float x",
        "property float y",
        "property float z",
        "property uchar intensity",
        "property ushort ring",
        "property float azimuth",
        "property float elevation",
        "property uchar return_id",
        "property uchar num_returns",
    ]
    # optional props in canonical order
    if have.get("range_m"):
        lines.append("property float range_m")
    if have.get("cos_incidence"):
        lines.append("property float cos_incidence")
    if have.get("mat_class"):
        lines.append("property uchar mat_class")
    if have.get("reflectivity"):
        lines.append("property float reflectivity")
    if have.get("transmittance"):
        lines.append("property float transmittance")
    if have_normals:
        lines += ["property float nx", "property float ny", "property float nz"]
    lines.append("end_header")
    return "\n".join(lines) + "\n"


def _stack_record_array(
    data: Dict,
) -> Tuple[np.ndarray, Dict[str, bool], Optional[np.ndarray]]:
    """Column‑stack core and optional attributes into a dense array for writing."""
    P = _coerce_points(data["points"])
    N = P.shape[0]

    cols = [P[:, 0], P[:, 1], P[:, 2]]

    # base
    arr = _coerce_col(data, "intensity", "u1", N)
    intensity = arr if arr is not None else np.zeros(N, "u1")
    arr = _coerce_col(data, "ring", "u2", N)
    ring = arr if arr is not None else np.zeros(N, "u2")
    arr = _coerce_col(data, "azimuth", "f4", N)
    az = arr if arr is not None else np.zeros(N, "f4")
    arr = _coerce_col(data, "elevation", "f4", N)
    el = arr if arr is not None else np.zeros(N, "f4")
    arr = _coerce_col(data, "return_id", "u1", N)
    rid = arr if arr is not None else np.ones(N, "u1")
    arr = _coerce_col(data, "num_returns", "u1", N)
    nret = arr if arr is not None else np.ones(N, "u1")

    cols += [intensity, ring, az, el, rid, nret]

    # optionals
    have = {}
    for k, dt in _OPT_FIELDS:
        arr_opt = _coerce_col(data, k, dt, N)
        have[k] = arr_opt is not None
        if arr_opt is not None:
            cols.append(arr_opt)

    normals = _detect_normals(data, N)
    if normals is not None:
        cols += [normals[:, 0], normals[:, 1], normals[:, 2]]

    rec = np.column_stack(cols)
    return rec, have, normals


def save_ply(
    path: str | Path, data: Dict[str, np.ndarray], binary: bool = False
) -> None:
    """Write a PLY with the fields produced by the LiDAR pipeline.

    Required: points (N,3). Optional fields include intensity, ring, azimuth,
    elevation, return_id, num_returns, range_m, cos_incidence, mat_class,
    reflectivity, transmittance, and normals (packed or split).
    """
    path = Path(path)
    if "points" not in data:
        raise ValueError("save_ply: 'points' (N,3) array is required")

    rec, have, normals = _stack_record_array(data)
    N = rec.shape[0]
    header = _build_header(N, have, normals is not None, binary)

    if not binary:
        # ASCII writer
        with path.open("w", encoding="utf-8") as fh:
            fh.write(header)
            # Write rows with exact number of columns
            for row in rec:
                # cast to python types to avoid numpy repr noise
                out = []
                # x,y,z floats
                out += [
                    f"{float(row[0]):.8f}",
                    f"{float(row[1]):.8f}",
                    f"{float(row[2]):.8f}",
                ]
                # intensity u8, ring u16
                out += [str(int(row[3])), str(int(row[4]))]
                # azimuth, elevation
                out += [f"{float(row[5]):.8f}", f"{float(row[6]):.8f}"]
                # return_id, num_returns
                out += [str(int(row[7])), str(int(row[8]))]
                # remaining columns as floats/ints as present
                for v in row[9:]:
                    # detect integer columns by close-to-integer dtype in layout decision
                    out.append(
                        str(float(v))
                        if isinstance(v, float) or np.issubdtype(type(v), np.floating)
                        else str(int(v))
                    )
                fh.write(" ".join(out) + "\n")
        return

    # Binary little-endian
    with path.open("wb") as fh:
        fh.write(header.encode("ascii"))
        # Build per-row struct format based on actual columns present
        fmt = "<"  # little-endian
        # x,y,z
        fmt += "fff"
        # intensity(u1), ring(u2)
        fmt += "BH"
        # azimuth, elevation
        fmt += "ff"
        # return_id, num_returns
        fmt += "BB"
        # optionals in canonical order
        if have.get("range_m"):
            fmt += "f"
        if have.get("cos_incidence"):
            fmt += "f"
        if have.get("mat_class"):
            fmt += "B"
        if have.get("reflectivity"):
            fmt += "f"
        if have.get("transmittance"):
            fmt += "f"
        if normals is not None:
            fmt += "fff"

        pack = struct.Struct(fmt).pack
        # Iterate rows; map types to python scalars
        for row in rec:
            vals = []
            # x,y,z
            vals += [float(row[0]), float(row[1]), float(row[2])]
            # intensity, ring
            vals += [int(row[3]) & 0xFF, int(row[4]) & 0xFFFF]
            # azimuth, elevation
            vals += [float(row[5]), float(row[6])]
            # return_id, num_returns
            vals += [int(row[7]) & 0xFF, int(row[8]) & 0xFF]
            # optionals
            c = 9
            if have.get("range_m"):
                vals.append(float(row[c]))
                c += 1
            if have.get("cos_incidence"):
                vals.append(float(row[c]))
                c += 1
            if have.get("mat_class"):
                vals.append(int(row[c]) & 0xFF)
                c += 1
            if have.get("reflectivity"):
                vals.append(float(row[c]))
                c += 1
            if have.get("transmittance"):
                vals.append(float(row[c]))
                c += 1
            if normals is not None:
                vals += [float(row[c]), float(row[c + 1]), float(row[c + 2])]
                c += 3
            fh.write(pack(*vals))
