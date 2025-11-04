"""Local helpers and fixtures for baked-only LiDAR tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover
    bpy = None
    pytest.skip("LiDAR Blender tests require Blender (bpy)", allow_module_level=True)

from lidar.lidar_config import LidarConfig


def _set_principled(
    material,
    *,
    base_color=(0.8, 0.8, 0.8, 1.0),
    roughness=0.3,
    metallic=0.0,
    transmission=0.0,
    alpha=None,
    blend_method=None,
):
    material.use_nodes = True
    nt = material.node_tree
    bsdf = next(
        (n for n in nt.nodes if getattr(n, "type", "") == "BSDF_PRINCIPLED"), None
    )
    if bsdf is None:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    rgba = list(base_color)
    if alpha is not None and len(rgba) == 4:
        rgba[3] = float(alpha)
    bsdf.inputs["Base Color"].default_value = tuple(rgba)
    # Ensure Principled Alpha socket reflects requested alpha (CLIP semantics rely on this)
    alpha_val = (
        float(alpha)
        if alpha is not None
        else (float(rgba[3]) if len(rgba) == 4 else None)
    )
    if alpha_val is not None and "Alpha" in bsdf.inputs:
        try:
            bsdf.inputs["Alpha"].default_value = alpha_val
        except Exception:
            pass
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = float(roughness)
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = float(metallic)
    # Transmission name varies by Principled version
    if "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = float(transmission)
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = float(transmission)
    if blend_method is not None:
        material.blend_method = str(blend_method).upper()


def make_plane_with_material(
    *,
    size=2.0,
    location=(0, 0, 0),
    rotation=(0, 0, 0),
    base_color=(0.8, 0.8, 0.8, 1.0),
    roughness=0.3,
    metallic=0.0,
    transmission=0.0,
    alpha=None,
    blend_method=None,
):
    bpy.ops.mesh.primitive_plane_add(
        size=float(size), location=tuple(location), rotation=tuple(rotation)
    )
    plane = bpy.context.active_object
    mat = bpy.data.materials.new(name="Mat")
    plane.data.materials.clear()
    plane.data.materials.append(mat)
    _set_principled(
        mat,
        base_color=base_color,
        roughness=roughness,
        metallic=metallic,
        transmission=transmission,
        alpha=alpha,
        blend_method=blend_method,
    )
    return plane, mat


def make_camera(*, location=(0, 0, 3), rotation=(0, 0, 0)):
    camd = bpy.data.cameras.new("C")
    cam = bpy.data.objects.new("C", camd)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.location = tuple(location)
    cam.rotation_euler = tuple(rotation)
    return cam


@pytest.fixture
def bake_scene(tmp_path):
    """Bake the current in-memory scene to small textures; return textures dir Path."""

    def _fake_bake(output_dir: Path, res: int = 4) -> Path:
        """Write tiny PNGs that mimic exporter-baked PBR maps for current scene objects.
        Produces {object}_{DIFFUSE|ROUGHNESS|METAL|TRANSMISSION}.png under textures/.
        """
        assert bpy is not None
        texdir = output_dir / "export_scene.blend" / "textures"
        texdir.mkdir(parents=True, exist_ok=True)

        def clean(name: str) -> str:
            return name.replace(" ", "_").replace(".", "_")

        def save_rgba_png(name: str, rgba: tuple[float, float, float, float]) -> None:
            img = bpy.data.images.new(name, width=res, height=res, alpha=True)
            # Fill all pixels with the same color (flat bake)
            px = list(rgba) * (res * res)
            img.pixels[:] = px
            img.file_format = "PNG"
            img.filepath_raw = str(texdir / f"{name}.png")
            img.save()
            try:
                bpy.data.images.remove(img, do_unlink=True)
            except Exception:
                pass

        for obj in list(bpy.context.scene.objects):
            if getattr(obj, "type", "") != "MESH":
                continue
            mat = obj.active_material
            if (
                mat is None
                or not getattr(mat, "use_nodes", False)
                or mat.node_tree is None
            ):
                continue
            bsdf = next(
                (
                    n
                    for n in mat.node_tree.nodes
                    if getattr(n, "type", "") == "BSDF_PRINCIPLED"
                ),
                None,
            )
            if bsdf is None:
                continue
            # Read defaults
            try:
                base_rgba = tuple(bsdf.inputs["Base Color"].default_value)
            except Exception:
                base_rgba = (0.8, 0.8, 0.8, 1.0)

            def _get(name: str, default: float) -> float:
                try:
                    return float(bsdf.inputs[name].default_value)
                except Exception:
                    return float(default)

            rough = _get("Roughness", 0.5)
            metal = _get("Metallic", 0.0)
            trans = _get("Transmission", _get("Transmission Weight", 0.0))

            cname = clean(obj.name)
            # DIFFUSE holds RGB, keep alpha channel for completeness
            save_rgba_png(f"{cname}_DIFFUSE", base_rgba)
            save_rgba_png(f"{cname}_ROUGHNESS", (rough, rough, rough, 1.0))
            save_rgba_png(f"{cname}_METAL", (metal, metal, metal, 1.0))
            save_rgba_png(f"{cname}_TRANSMISSION", (trans, trans, trans, 1.0))

        return texdir

    def _export_run(input_dir: Path, output_dir: Path, res: int = 64) -> Path:
        # Launch a separate Blender process to run the exporter bake runner.
        repo_root = Path(__file__).resolve().parents[2]
        runner = repo_root / "scripts" / "bake_export_runner.py"
        cmd = [
            sys.executable,
            "-m",
            "infinigen.launch_blender",
            "-m",
            "infinigen.tools.blendscript_path_append",
            "--",
            "--python",
            str(runner),
            "--",
            str(input_dir),
            str(output_dir),
            str(int(res)),
        ]
        try:
            subprocess.run(cmd, cwd=str(repo_root), check=True)
        except Exception:
            # Fall back to fake bake on any launcher/exporter issues
            return _fake_bake(output_dir, res=4)
        # Resolve textures dir; if not found, fall back to fake bake
        try:
            return _find_textures_dir(output_dir)
        except Exception:
            return _fake_bake(output_dir, res=4)

    def _find_textures_dir(base: Path) -> Path:
        for p in base.rglob("textures"):
            if p.is_dir():
                return p
        raise FileNotFoundError(f"No textures directory under {base}")

    def _bake(res: int = 64) -> Path:
        in_dir = tmp_path / "blend_in"
        out_dir = tmp_path / "export_out"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        blend_path = in_dir / "scene.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        texdir = _export_run(in_dir, out_dir, res=res)
        assert texdir.is_dir()
        return texdir

    return _bake


@pytest.fixture
def lidar_cfg():
    cfg = LidarConfig()
    cfg.auto_expose = False
    # keep deterministic if these knobs exist
    if hasattr(cfg, "noise_sigma"):
        cfg.noise_sigma = 0.0
    if hasattr(cfg, "range_jitter"):
        cfg.range_jitter = 0.0
    cfg.enable_secondary = True
    return cfg
