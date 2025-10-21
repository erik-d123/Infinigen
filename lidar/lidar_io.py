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

def save_ply(output_dir, frame, xform_world_to_out: Matrix, data, cfg):
    # Save point cloud in PLY format (ASCII)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"lidar_frame_{frame:04d}.ply")

    # Transform points to output coordinate frame
    pts_out = np.array([(xform_world_to_out @ Vector(p))[:] for p in data["points_world"]], dtype=np.float32)
    n = len(pts_out)
    
    # Check for optional fields
    has_ci = "cos_incidence" in data
    has_mc = "mat_class" in data
    has_power = "return_power" in data
    has_range = "range_m" in data
    has_normals = "normals_world" in data and n == len(data["normals_world"])
    if has_normals:
        rot = _matrix_to_np3x3(xform_world_to_out.to_3x3())
        normals_out = (rot @ np.asarray(data["normals_world"], dtype=np.float32).T).T

    binary = getattr(cfg, "ply_binary", False)
    if binary:
        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            binary = False

    if binary:
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
                 ("intensity", "u1"), ("ring", "u2"),
                 ("azimuth", "f4"), ("elevation", "f4"), ("time_offset", "f4"),
                 ("return_id", "u1"), ("num_returns", "u1")]
        if has_range:
            dtype.append(("range_m", "f4"))
        if has_ci:
            dtype.append(("cos_incidence", "f4"))
        if has_mc:
            dtype.append(("mat_class", "u1"))
        if has_power:
            dtype.append(("return_power", "f4"))
        if "exposure_scale" in data:
            dtype.append(("exposure_scale", "f4"))
        if has_normals:
            dtype.extend([("nx", "f4"), ("ny", "f4"), ("nz", "f4")])
        vertex = np.empty(n, dtype=dtype)
        vertex["x"] = pts_out[:, 0]
        vertex["y"] = pts_out[:, 1]
        vertex["z"] = pts_out[:, 2]
        vertex["intensity"] = np.asarray(data["intensities_u8"], dtype=np.uint8)
        vertex["ring"] = np.asarray(data["ring_ids"], dtype=np.uint16)
        vertex["azimuth"] = np.asarray(data["azimuth_rad"], dtype=np.float32)
        vertex["elevation"] = np.asarray(data["elevation_rad"], dtype=np.float32)
        vertex["time_offset"] = np.asarray(data["time_offset"], dtype=np.float32)
        vertex["return_id"] = np.asarray(data["return_id"], dtype=np.uint8)
        vertex["num_returns"] = np.asarray(data["num_returns"], dtype=np.uint8)
        if has_range:
            vertex["range_m"] = np.asarray(data["range_m"], dtype=np.float32)
        if has_ci:
            vertex["cos_incidence"] = np.asarray(data["cos_incidence"], dtype=np.float32)
        if has_mc:
            vertex["mat_class"] = np.asarray(data["mat_class"], dtype=np.uint8)
        if has_power:
            power_data = data.get("return_power", data.get("reflectance"))
            vertex["return_power"] = np.asarray(power_data, dtype=np.float32)
        if "exposure_scale" in data:
            vertex["exposure_scale"] = np.asarray(data["exposure_scale"], dtype=np.float32)
        if has_normals:
            vertex["nx"] = normals_out[:, 0]
            vertex["ny"] = normals_out[:, 1]
            vertex["nz"] = normals_out[:, 2]
        PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)
        return

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"ply\nformat ascii 1.0\ncomment Lidar frame {frame}\n")
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
        if "exposure_scale" in data:
            f.write("property float exposure_scale\n")
        if has_normals:
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("end_header\n")

        az, el, t = data["azimuth_rad"], data["elevation_rad"], data["time_offset"]
        ret, nret = data["return_id"], data["num_returns"]
        ci = data.get("cos_incidence", None)
        mc = data.get("mat_class", None)
        power = data.get("return_power", None)
        rng = data.get("range_m", None)
        exposure = data.get("exposure_scale", None)

        for i in range(n):
            x, y, z = pts_out[i]
            line = (f"{x:.6f} {y:.6f} {z:.6f} {int(data['intensities_u8'][i])} {int(data['ring_ids'][i])} "
                    f"{az[i]:.6f} {el[i]:.6f} {t[i]:.6e} {int(ret[i])} {int(nret[i])}")
            if has_range:
                line += f" {rng[i]:.6f}"
            if has_ci:
                line += f" {ci[i]:.6f}"
            if has_mc:
                line += f" {int(mc[i])}"
            if has_power:
                line += f" {power[i]:.6f}"
            if exposure is not None:
                line += f" {exposure[i]:.6f}"
            if has_normals:
                nx, ny, nz = normals_out[i]
                line += f" {nx:.6f} {ny:.6f} {nz:.6f}"
            f.write(line + "\n")
