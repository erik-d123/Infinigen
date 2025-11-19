# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Principled-first material sampling for LiDAR.

This sampler inspects the active Principled BSDF for a hit material, evaluates the
inputs (Base Color, Roughness, Metallic, Specular/IOR, Transmission, etc.) at the
hit UV, and returns the per-hit properties needed by the intensity model.

It supports three kinds of sources per Principled input:
 1. Constant/unlinked sockets (use default_value)
 2. Image Texture nodes (sampled bilinearly at UV)
 3. Arbitrary node graphs (baked once to an internal image via EMIT bake)

When the Principled workflow cannot be evaluated (e.g., no Principled node), the caller
should treat it as an error; no baked-texture fallback is provided here.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import bpy
except ImportError:
    bpy = None

try:
    from mathutils import Vector
except ImportError:
    Vector = None


def _compute_barycentric(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> Tuple[float, float, float]:
    """Return barycentric coordinates of point p in triangle (a, b, c)."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return (1.0, 0.0, 0.0)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return (float(u), float(v), float(w))


def _hit_uv(
    eval_obj, mesh, poly_index: int, hit_world
) -> Optional[Tuple[float, float]]:
    """Compute active UV at a world‑space hit on a specific polygon.

    Returns (u, v) in [0, 1] if successful, otherwise None.
    """
    uv_layer = getattr(mesh.uv_layers, "active", None)
    if uv_layer is None:
        return None
    uv_data = uv_layer.data
    try:
        M_inv = eval_obj.matrix_world.inverted()
        hit_local_v = M_inv @ Vector(hit_world)
        hit_local = np.array(hit_local_v, dtype=np.float64)
    except Exception:
        return None
    mesh.calc_loop_triangles()
    for tri in mesh.loop_triangles:
        if tri.polygon_index != poly_index:
            continue
        l0, l1, l2 = tri.loops
        vi0 = mesh.loops[l0].vertex_index
        vi1 = mesh.loops[l1].vertex_index
        vi2 = mesh.loops[l2].vertex_index
        a = np.array(mesh.vertices[vi0].co, dtype=np.float64)
        b = np.array(mesh.vertices[vi1].co, dtype=np.float64)
        c = np.array(mesh.vertices[vi2].co, dtype=np.float64)
        u, v, w = _compute_barycentric(hit_local, a, b, c)
        uv0 = np.array(uv_data[l0].uv, dtype=np.float64)
        uv1 = np.array(uv_data[l1].uv, dtype=np.float64)
        uv2 = np.array(uv_data[l2].uv, dtype=np.float64)
        uv = u * uv0 + v * uv1 + w * uv2
        return (float(uv[0]), float(uv[1]))
    return None


class PrincipledSampleError(RuntimeError):
    """Raised when a Principled material cannot be sampled."""


def _clean(name: str) -> str:
    return name.replace(" ", "_").replace(".", "_")


def _to_rgba(val) -> Tuple[float, float, float, float]:
    """Robustly coerce Blender socket default (bpy_prop_array, scalars) to RGBA floats."""
    try:
        seq = val[:]
        vals = [float(x) for x in seq]
    except Exception:
        try:
            seq = list(val)
            flat = []
            for x in seq:
                try:
                    flat.extend(list(x[:]))
                except Exception:
                    flat.append(x)
            vals = [float(x) for x in flat]
        except Exception:
            try:
                v = float(val)
            except Exception:
                v = 0.0
            vals = [v]
    if len(vals) >= 4:
        return (vals[0], vals[1], vals[2], vals[3])
    if len(vals) == 3:
        return (vals[0], vals[1], vals[2], 1.0)
    if len(vals) == 2:
        return (vals[0], vals[1], vals[0], 1.0)
    if len(vals) == 1:
        return (vals[0], vals[0], vals[0], 1.0)
    return (0.0, 0.0, 0.0, 1.0)


def _bilinear(px: np.ndarray, uv: Tuple[float, float]) -> np.ndarray:
    """Sample an HxWx4 image array at UV in [0,1) with bilinear filtering."""
    h, w, _ = px.shape
    if w <= 0 or h <= 0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    u = float(uv[0]) % 1.0
    v = float(uv[1]) % 1.0
    x = u * (w - 1)
    y = v * (h - 1)
    x0 = int(math.floor(x))
    x1 = min(w - 1, x0 + 1)
    y0 = int(math.floor(y))
    y1 = min(h - 1, y0 + 1)
    dx = x - x0
    dy = y - y0
    c00 = px[y0, x0]
    c10 = px[y0, x1]
    c01 = px[y1, x0]
    c11 = px[y1, x1]
    c0 = c00 * (1.0 - dx) + c10 * dx
    c1 = c01 * (1.0 - dx) + c11 * dx
    c = c0 * (1.0 - dy) + c1 * dy
    return c


@dataclasses.dataclass
class _BakedInput:
    array: np.ndarray
    width: int
    height: int


@dataclasses.dataclass
class _NodeContext:
    tree: bpy.types.NodeTree
    node: bpy.types.ShaderNode


_PrincipledContext = _NodeContext


class PrincipledSampler:
    """Singleton sampler that extracts Principled inputs per hit."""

    _inst: Optional["PrincipledSampler"] = None

    def __init__(self, bake_resolution: int = 1024):
        self.bake_resolution = bake_resolution
        self.image_cache: Dict[str, np.ndarray] = {}
        self.bake_cache: Dict[Tuple[str, str, str], _BakedInput] = {}
        self.mesh_cache: Dict[str, Tuple[bpy.types.Object, bpy.types.Mesh, int]] = {}
        self.mesh_epoch: Optional[int] = None

    @classmethod
    def get(cls) -> "PrincipledSampler":
        if cls._inst is None:
            cls._inst = PrincipledSampler()
        return cls._inst

    def _ensure_mesh_epoch(self, frame_idx: int) -> None:
        if self.mesh_epoch is None:
            self.mesh_epoch = frame_idx
            return
        if self.mesh_epoch != frame_idx:
            try:
                for eval_obj, mesh, _ in self.mesh_cache.values():
                    with contextlib.suppress(Exception):
                        eval_obj.to_mesh_clear()  # type: ignore[attr-defined]
            finally:
                self.mesh_cache.clear()
                self.mesh_epoch = frame_idx

    def sample(
        self,
        obj,
        depsgraph,
        poly_index: int,
        hit_world,
        cfg=None,
    ) -> Dict:
        """Sample Principled inputs for the hit."""
        if bpy is None:
            raise PrincipledSampleError("Principled sampling requires bpy")

        if cfg is not None and hasattr(cfg, "principled_bake_res"):
            self.bake_resolution = max(
                32, int(getattr(cfg, "principled_bake_res", self.bake_resolution))
            )

        try:
            frame_idx = int(bpy.context.scene.frame_current)
        except Exception:
            frame_idx = -1
        self._ensure_mesh_epoch(frame_idx)

        obj_key = getattr(obj, "name", None) or str(id(obj))
        if obj_key in self.mesh_cache:
            eval_obj, eval_mesh, _ = self.mesh_cache[obj_key]
        else:
            eval_obj = obj.evaluated_get(depsgraph)
            eval_mesh = eval_obj.to_mesh()
            self.mesh_cache[obj_key] = (eval_obj, eval_mesh, frame_idx)

        try:
            uv = _hit_uv(eval_obj, eval_mesh, poly_index, hit_world)
        except Exception:
            uv = None
        if uv is None or not getattr(eval_mesh, "uv_layers", None):
            # Ensure the source object has a UV map; then refresh evaluated mesh and retry
            self._ensure_uv_map(obj)
            try:
                bpy.context.view_layer.update()
            except Exception:
                pass
            eval_obj = obj.evaluated_get(depsgraph)
            eval_mesh = eval_obj.to_mesh()
            try:
                uv = _hit_uv(eval_obj, eval_mesh, poly_index, hit_world)
            except Exception:
                uv = None
        if uv is None:
            raise PrincipledSampleError("UV sampling failed")

        poly = eval_mesh.polygons[poly_index]
        mat_idx = int(getattr(poly, "material_index", 0))
        mat = None
        if eval_obj.material_slots and mat_idx < len(eval_obj.material_slots):
            mat = eval_obj.material_slots[mat_idx].material
        if mat is None and obj.material_slots and mat_idx < len(obj.material_slots):
            mat = obj.material_slots[mat_idx].material
        if mat is None:
            mat = eval_obj.active_material or obj.active_material
        if mat is None:
            raise PrincipledSampleError("Polygon has no material")

        context = self._find_principled_context(mat)
        if context is None:
            sampled = self._sample_non_principled(mat, uv, obj)
            if sampled is None:
                raise PrincipledSampleError(f"No Principled BSDF on {mat.name}")
            return sampled

        base_rgba = self._sample_color(context, "Base Color", uv, obj)
        roughness = self._sample_scalar(context, "Roughness", uv, obj)
        metallic = self._sample_scalar(context, "Metallic", uv, obj)
        transmission = self._sample_scalar(context, "Transmission", uv, obj)
        trans_rough = self._sample_scalar(
            context, "Transmission Roughness", uv, obj, default=0.0
        )

        alpha_socket_linked = False
        try:
            alpha_socket_linked = bool(context.node.inputs.get("Alpha").is_linked)
        except Exception:
            alpha_socket_linked = False
        alpha_val = self._sample_scalar(
            context,
            "Alpha",
            uv,
            obj,
            default=base_rgba[3] if len(base_rgba) > 3 else 1.0,
            force_bake=True,
        )
        coverage = float(max(0.0, min(1.0, alpha_val)))
        alpha_mode = str(getattr(mat, "blend_method", "BLEND")).upper()
        alpha_clip = float(getattr(mat, "alpha_threshold", 0.5) or 0.5)

        sample: Dict = {
            "base_color": tuple(float(c) for c in base_rgba[:3]),
            "metallic": float(metallic),
            "roughness": float(roughness),
            "transmission": float(transmission),
            "transmission_roughness": float(trans_rough),
            "coverage": coverage,
            "alpha_linked": bool(alpha_socket_linked),
            "alpha_mode": alpha_mode,
            "alpha_clip": alpha_clip,
        }

        principled = context.node
        if "IOR" in principled.inputs and principled.inputs["IOR"].enabled:
            sample["ior"] = float(
                self._sample_scalar(context, "IOR", uv, obj, default=1.45)
            )
        else:
            # Principled Specular is defined so that Specular=0.5 → F0≈0.04
            spec = self._sample_scalar(context, "Specular", uv, obj, default=0.5)
            sample["specular"] = float(spec)

        return sample

    # ---------------------- node utilities ----------------------

    def _find_principled_context(self, mat) -> Optional[_NodeContext]:
        return self._find_node_context(mat, {"BSDF_PRINCIPLED"})

    def _find_node_context(self, mat, node_types: set[str]) -> Optional[_NodeContext]:
        if not (mat and getattr(mat, "use_nodes", False) and mat.node_tree):
            return None
        visited: set[int] = set()
        return self._find_in_tree(mat.node_tree, visited, node_types)

    def _find_in_tree(
        self, tree: bpy.types.NodeTree, visited: set[int], node_types: set[str]
    ) -> Optional[_NodeContext]:
        if tree is None:
            return None
        key = int(tree.as_pointer())
        if key in visited:
            return None
        visited.add(key)
        for node in tree.nodes:
            if node.type in node_types:
                return _NodeContext(tree=tree, node=node)
        for node in tree.nodes:
            if node.type == "GROUP" and getattr(node, "node_tree", None):
                ctx = self._find_in_tree(node.node_tree, visited, node_types)
                if ctx is not None:
                    return ctx
        return None

    def _ensure_uv_map(self, obj) -> None:
        try:
            if not getattr(obj, "data", None):
                return
            uv_layers = getattr(obj.data, "uv_layers", None)
            if uv_layers and len(uv_layers) > 0:
                return
            # Create and unwrap a UV map
            # Save selection & active
            layer = bpy.context.view_layer
            prev_active = layer.objects.active
            # Deselect all
            with contextlib.suppress(Exception):
                bpy.ops.object.select_all(action="DESELECT")
            with contextlib.suppress(Exception):
                obj.select_set(True)
            layer.objects.active = obj
            # Ensure OBJECT mode
            with contextlib.suppress(Exception):
                bpy.ops.object.mode_set(mode="OBJECT")
            # Add UV layer
            with contextlib.suppress(Exception):
                obj.data.uv_layers.new(name="LiDARUV")
                obj.data.uv_layers.active = obj.data.uv_layers[-1]
            # Smart unwrap
            with contextlib.suppress(Exception):
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.uv.smart_project(angle_limit=0.7)
                bpy.ops.object.mode_set(mode="OBJECT")
            # Restore selection
            if prev_active is not None:
                layer.objects.active = prev_active
        except Exception:
            pass

    def _sample_non_principled(self, mat, uv, obj) -> Optional[Dict]:
        # Known simple shader types
        glass_ctx = self._find_node_context(mat, {"BSDF_GLASS"})
        if glass_ctx is not None:
            return self._sample_glass(glass_ctx, mat, uv, obj)

        diffuse_ctx = self._find_node_context(mat, {"BSDF_DIFFUSE"})
        if diffuse_ctx is not None:
            return self._sample_diffuse(diffuse_ctx, mat, uv, obj)

        glossy_ctx = self._find_node_context(mat, {"BSDF_GLOSSY"})
        if glossy_ctx is not None:
            return self._sample_glossy(glossy_ctx, mat, uv, obj)

        transparent_ctx = self._find_node_context(mat, {"BSDF_TRANSPARENT"})
        if transparent_ctx is not None:
            return self._sample_transparent(transparent_ctx, mat, uv, obj)

        return None

    def _alpha_settings(self, mat) -> Tuple[str, float]:
        alpha_mode = str(getattr(mat, "blend_method", "BLEND")).upper()
        try:
            alpha_clip = float(getattr(mat, "alpha_threshold", 0.5) or 0.5)
        except Exception:
            alpha_clip = 0.5
        return alpha_mode, alpha_clip

    def _sample_glass(self, ctx, mat, uv, obj) -> Dict:
        color = self._sample_color(ctx, "Color", uv, obj)
        rough = self._sample_scalar(ctx, "Roughness", uv, obj, default=0.0)
        ior = self._sample_scalar(ctx, "IOR", uv, obj, default=1.45)
        alpha_mode, alpha_clip = self._alpha_settings(mat)
        return {
            "base_color": tuple(float(c) for c in color[:3]),
            "metallic": 0.0,
            "roughness": float(rough),
            "transmission": 1.0,
            "transmission_roughness": float(rough),
            "coverage": 1.0,
            "alpha_mode": alpha_mode,
            "alpha_clip": alpha_clip,
            "ior": float(ior),
        }

    def _sample_diffuse(self, ctx, mat, uv, obj) -> Dict:
        color = self._sample_color(ctx, "Color", uv, obj)
        rough = self._sample_scalar(ctx, "Roughness", uv, obj, default=0.5)
        alpha_mode, alpha_clip = self._alpha_settings(mat)
        return {
            "base_color": tuple(float(c) for c in color[:3]),
            "metallic": 0.0,
            "roughness": float(rough),
            "transmission": 0.0,
            "transmission_roughness": 0.0,
            "coverage": 1.0,
            "alpha_mode": alpha_mode,
            "alpha_clip": alpha_clip,
            "specular": 0.5,
        }

    def _sample_glossy(self, ctx, mat, uv, obj) -> Dict:
        color = self._sample_color(ctx, "Color", uv, obj)
        rough = self._sample_scalar(ctx, "Roughness", uv, obj, default=0.05)
        alpha_mode, alpha_clip = self._alpha_settings(mat)
        return {
            "base_color": tuple(float(c) for c in color[:3]),
            "metallic": 1.0,
            "roughness": float(rough),
            "transmission": 0.0,
            "transmission_roughness": 0.0,
            "coverage": 1.0,
            "alpha_mode": alpha_mode,
            "alpha_clip": alpha_clip,
            "specular": 1.0,
        }

    def _sample_transparent(self, ctx, mat, uv, obj) -> Dict:
        color = self._sample_color(ctx, "Color", uv, obj)
        alpha_mode, alpha_clip = self._alpha_settings(mat)
        return {
            "base_color": tuple(float(c) for c in color[:3]),
            "metallic": 0.0,
            "roughness": 0.0,
            "transmission": 1.0,
            "transmission_roughness": 0.0,
            "coverage": 1.0,
            "alpha_mode": alpha_mode,
            "alpha_clip": alpha_clip,
            "specular": 0.5,
        }

    def _sample_color(
        self,
        context: _PrincipledContext,
        socket_name: str,
        uv,
        obj,
        default=(1.0, 1.0, 1.0, 1.0),
    ):
        baked = self._bake_input(obj, context, socket_name)
        if baked is not None:
            rgba = _bilinear(baked.array, uv)
            return rgba
        return default

    def _sample_scalar(
        self,
        context: _PrincipledContext,
        socket_name: str,
        uv,
        obj,
        default: float = 0.0,
        force_bake: bool = False,
    ) -> float:
        arr = self._sample_socket(
            context, socket_name, uv, obj, is_color=False, force_bake=force_bake
        )
        if arr is None:
            return default
        if isinstance(arr, (tuple, list, np.ndarray)):
            return float(arr[0])
        return float(arr)

    def _sample_socket(
        self,
        context: _PrincipledContext,
        socket_name: str,
        uv,
        obj,
        is_color: bool,
        force_bake: bool = False,
    ):
        principled = context.node
        if socket_name not in principled.inputs:
            return None
        socket = principled.inputs[socket_name]
        if force_bake or is_color:
            baked = self._bake_input(obj, context, socket_name)
            if baked is not None:
                rgba = _bilinear(baked.array, uv)
                return rgba if is_color else rgba[0]
        if socket.is_linked and socket.links and not force_bake:
            img = self._trace_image(socket.links[0].from_node)
            if img is not None:
                arr = self._image_pixels(img)
                rgba = _bilinear(arr, uv)
                return rgba if is_color else rgba[0]
            # Fallback to baking this input once
            baked = self._bake_input(obj, context, socket_name)
            if baked is not None:
                rgba = _bilinear(baked.array, uv)
                return rgba if is_color else rgba[0]
        # Unlinked: use default value directly
        default = getattr(socket, "default_value", None)
        if default is None:
            return None
        if is_color and hasattr(default, "__len__"):
            return tuple(float(c) for c in default)
        return float(default)

    def _trace_image(self, node):
        """Return a Blender image if node (or its ancestors) is a Tex Image."""
        if node is None:
            return None
        if node.type == "TEX_IMAGE":
            return node.image
        # Simple passthrough nodes
        passthrough = {"MAPPING", "TEX_COORD", "NORMALIZE", "VECT_TRANSFORM"}
        if node.type in passthrough and node.inputs:
            # Choose first input that is linked
            for inp in node.inputs:
                if inp.is_linked and inp.links:
                    return self._trace_image(inp.links[0].from_node)
        return None

    def _image_pixels(self, image):
        if image is None:
            return None
        key = f"{image.library_filepath}:{image.name_full}"
        if key in self.image_cache:
            return self.image_cache[key]
        w, h = image.size
        arr = np.array(image.pixels[:], dtype=np.float32).reshape((h, w, 4)).copy()
        self.image_cache[key] = arr
        return arr

    # ------------------ baking fallback ------------------

    def _bake_input(
        self, obj, context: _PrincipledContext, socket_name: str
    ) -> Optional[_BakedInput]:
        mesh_name = getattr(getattr(obj, "data", None), "name", "")
        mat_name = (
            getattr(obj.active_material, "name", "") if obj.active_material else ""
        )
        tree_name = getattr(context.tree, "name", "")
        key = (mesh_name, mat_name, tree_name, context.node.name, socket_name)
        if key in self.bake_cache:
            return self.bake_cache[key]

        scene = bpy.context.scene
        prev_engine = getattr(scene.render, "engine", "CYCLES")
        try:
            scene.render.engine = "CYCLES"
        except Exception:
            pass

        if bpy.context.mode != "OBJECT":
            with contextlib.suppress(Exception):
                bpy.ops.object.mode_set(mode="OBJECT")

        tree = context.tree
        nodes = tree.nodes
        links = tree.links
        principled = context.node
        socket = principled.inputs.get(socket_name)
        if socket is None:
            return None

        # Ensure we target the root material tree for the bake image,
        # otherwise "No active image found" if inside a group.
        root_tree = tree
        if obj.active_material and obj.active_material.node_tree:
            root_tree = obj.active_material.node_tree

        tree_name = getattr(tree, "name", "NodeTree")
        img = bpy.data.images.new(
            f"LiDAR_Principled_{_clean(tree_name)}_{socket_name}",
            width=self.bake_resolution,
            height=self.bake_resolution,
            alpha=True,
        )

        # Create texture node in the root tree so bake operator sees it
        tex_node = root_tree.nodes.new("ShaderNodeTexImage")
        tex_node.image = img
        root_tree.nodes.active = tex_node

        # Emission node stays in the context tree (where the signal is)
        emiss = nodes.new("ShaderNodeEmission")
        emiss.inputs["Strength"].default_value = 1.0

        created_links = []
        temp_nodes = []

        if socket.is_linked and socket.links:
            src = socket.links[0].from_socket
            created_links.append(links.new(src, emiss.inputs["Color"]))
        else:
            default = getattr(socket, "default_value", None)
            rgb = nodes.new("ShaderNodeRGB")
            temp_nodes.append(rgb)
            tup = _to_rgba(default)
            rgb.outputs[0].default_value = tup  # type: ignore[arg-type]
            created_links.append(links.new(rgb.outputs[0], emiss.inputs["Color"]))

        # Disconnect Principled output links so emission can drive same sockets
        removed_links = []
        bsdf_output = principled.outputs[0] if principled.outputs else None
        if bsdf_output and bsdf_output.links:
            for link in list(bsdf_output.links):
                from_sock = link.from_socket
                to_sock = link.to_socket
                removed_links.append((from_sock, to_sock))
                links.remove(link)
                created_links.append(links.new(emiss.outputs[0], to_sock))
        else:
            # Fallback: link emission to first available Material/Group output
            fallback_target = None
            for node in nodes:
                if node.type == "OUTPUT_MATERIAL" and "Surface" in node.inputs:
                    fallback_target = node.inputs["Surface"]
                    break
                if node.type == "GROUP_OUTPUT" and node.inputs:
                    fallback_target = node.inputs[0]
                    break
            if fallback_target is not None:
                created_links.append(links.new(emiss.outputs[0], fallback_target))
            else:
                nodes.remove(emiss)
                root_tree.nodes.remove(tex_node)
                bpy.data.images.remove(img, do_unlink=True)
                with contextlib.suppress(Exception):
                    scene.render.engine = prev_engine
                return None

        try:
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.bake(
                type="EMIT", save_mode="INTERNAL", use_clear=True, margin=2
            )
        except Exception as exc:
            # Cleanup and propagate
            baked = None
            err = exc
        else:
            arr = np.array(img.pixels[:], dtype=np.float32).reshape(
                (img.size[1], img.size[0], 4)
            )
            baked = _BakedInput(array=arr.copy(), width=img.size[0], height=img.size[1])
            err = None
        finally:
            # Restore links
            for l in created_links:
                with contextlib.suppress(Exception):
                    links.remove(l)
            for from_soc, to_soc in removed_links:
                with contextlib.suppress(Exception):
                    links.new(from_soc, to_soc)
            with contextlib.suppress(Exception):
                nodes.remove(emiss)
            with contextlib.suppress(Exception):
                root_tree.nodes.remove(tex_node)
            for node in temp_nodes:
                with contextlib.suppress(Exception):
                    nodes.remove(node)
            with contextlib.suppress(Exception):
                bpy.data.images.remove(img, do_unlink=True)
            with contextlib.suppress(Exception):
                scene.render.engine = prev_engine

        if err is not None:
            raise PrincipledSampleError(str(err))
        if baked is not None:
            self.bake_cache[key] = baked
        return baked
