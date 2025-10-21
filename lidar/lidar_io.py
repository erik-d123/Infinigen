#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# LiDAR I/O module
# Handles PLY file output and coordinate transformations

import os
import numpy as np
from mathutils import Matrix, Vector

def _matrix_to_np3x3(m: Matrix) -> np.ndarray:
    # Convert Blender Matrix to 3x3 numpy array
    return np.array([[m[0][0], m[0][1], m[0][2]],
                     [m[1][0], m[1][1], m[1][2]],
                     [m[2][0], m[2][1], m[2][2]]], dtype=np.float32)

def world_to_frame_matrix(cam_obj, sensor_R: Matrix, frame: str) -> Matrix:
    # Calculate transformation matrix for output coordinate frame
    if frame == 'world':
        return Matrix.Identity(4)
    
    cam_inv = cam_obj.matrix_world.inverted()
    if frame == 'camera':
        return cam_inv
    elif frame == 'sensor':
        # world->sensor = (sensor_R^{-1}) * (world->camera)
        R_sc = sensor_R.inverted().to_4x4()
        return R_sc @ cam_inv
    else:
        raise ValueError("ply_frame must be one of {'camera','sensor','world'}")

# PLY file output

def save_ply(output_dir, frame, xform_world_to_out: Matrix, data, cfg, *, binary: bool = None):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"lidar_frame_{frame:04d}.ply")

    pts_out = np.array([(xform_world_to_out @ Vector(p))[:] for p in data["points_world"]], dtype=np.float32)
    n = len(pts_out)

    has_ci = "cos_incidence" in data
    has_mc = "mat_class" in data
    has_power = "return_power" in data
    has_range = "range_m" in data
    trans_data = None
    if "transmittance" in data:
        trans_data = np.asarray(data["transmittance"], dtype=np.float32)
    elif "exposure_scale" in data:
        trans_data = np.asarray(data["exposure_scale"], dtype=np.float32)
    has_trans = trans_data is not None
    has_normals = "normals_world" in data and n == len(data["normals_world"])
    if has_normals:
        rot = _matrix_to_np3x3(xform_world_to_out.to_3x3())
        normals_out = (rot @ np.asarray(data["normals_world"], dtype=np.float32).T).T

    if binary is None:
        binary = getattr(cfg, "binary_ply", False)

    if binary:
        import struct
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"comment Lidar frame {frame}",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar intensity",
            "property ushort ring",
            "property float azimuth",
            "property float elevation",
            "property float time_offset",
            "property uchar return_id",
            "property uchar num_returns",
        ]
        if has_range:
            header_lines.append("property float range_m")
        if has_ci:
            header_lines.append("property float cos_incidence")
        if has_mc:
            header_lines.append("property uchar mat_class")
        if has_power:
            header_lines.append("property float return_power")
        if has_trans:
            header_lines.append("property float transmittance")
        if has_normals:
            header_lines.extend(["property float nx", "property float ny", "property float nz"])
        header_lines.append("end_header\n")

        with open(path, "wb") as f:
            for line in header_lines:
                f.write((line + "\n").encode("ascii"))
            for i in range(n):
                f.write(struct.pack("<fff", float(pts_out[i, 0]), float(pts_out[i, 1]), float(pts_out[i, 2])))
                f.write(struct.pack("<B", int(data["intensities_u8"][i])))
                f.write(struct.pack("<H", int(data["ring_ids"][i])))
                f.write(struct.pack("<fff", float(data["azimuth_rad"][i]), float(data["elevation_rad"][i]), float(data["time_offset"][i])))
                f.write(struct.pack("<B", int(data["return_id"][i])))
                f.write(struct.pack("<B", int(data["num_returns"][i])))
                if has_range:
                    f.write(struct.pack("<f", float(data["range_m"][i])))
                if has_ci:
                    f.write(struct.pack("<f", float(data["cos_incidence"][i])))
                if has_mc:
                    f.write(struct.pack("<B", int(data["mat_class"][i])))
                if has_power:
                    f.write(struct.pack("<f", float(data["return_power"][i])))
                if has_trans:
                    f.write(struct.pack("<f", float(trans_data[i])))
                if has_normals:
                    nx, ny, nz = normals_out[i]
                    f.write(struct.pack("<fff", float(nx), float(ny), float(nz)))
        return

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment Lidar frame {frame}\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar intensity\nproperty ushort ring\n")
        f.write("property float azimuth\nproperty float elevation\nproperty float time_offset\n")
        f.write("property uchar return_id\nproperty uchar num_returns\n")
        if has_range:
            f.write("property float range_m\n")
        if has_ci:
            f.write("property float cos_incidence\n")
        if has_mc:
            f.write("property uchar mat_class\n")
        if has_power:
            f.write("property float return_power\n")
        if has_trans:
            f.write("property float transmittance\n")
        if has_normals:
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("end_header\n")

        for i in range(n):
            line = (
                f"{pts_out[i,0]:.6f} {pts_out[i,1]:.6f} {pts_out[i,2]:.6f} "
                f"{int(data['intensities_u8'][i])} {int(data['ring_ids'][i])} "
                f"{data['azimuth_rad'][i]:.6f} {data['elevation_rad'][i]:.6f} {data['time_offset'][i]:.6f} "
                f"{int(data['return_id'][i])} {int(data['num_returns'][i])}"
            )
            if has_range:
                line += f" {data['range_m'][i]:.6f}"
            if has_ci:
                line += f" {data['cos_incidence'][i]:.6f}"
            if has_mc:
                line += f" {int(data['mat_class'][i])}"
            if has_power:
                line += f" {data['return_power'][i]:.6f}"
            if has_trans:
                line += f" {trans_data[i]:.6f}"
            if has_normals:
                nx, ny, nz = normals_out[i]
                line += f" {nx:.6f} {ny:.6f} {nz:.6f}"
            f.write(line + "\n")
