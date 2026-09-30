"""HSSD-specific helpers for gen_simchange.py (run inside bpy): interior lamps from tagged objects, application of a
rearrangement plan, object-index pass for change quantification."""
from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def _bbox(o):
    pts = [o.matrix_world @ Vector(c) for c in o.bound_box]
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return mn, mx


def lamp_lights(scene, energy: float, kelvin_rgb, max_lamps: int = 400):
    """Point lights at every object tagged hssd_lamp (ceiling lamps hang from the ceiling: light just below the mesh)."""
    made = []
    for o in list(scene.objects):
        if not o.get("hssd_lamp") or o.hide_render:
            continue
        mn, mx = _bbox(o)
        c = (mn + mx) / 2
        cat = o.get("hssd_category", "")
        if cat in ("ceiling_lamp", "chandelier", "pendant_lamp"):
            pos = Vector((c.x, c.y, mn.z - 0.12))
            e = energy
        elif cat == "wall_lamp":
            # push the light away from the wall along the object's local front (+Y after the Y-up conversion)
            fwd = (o.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized()
            pos = c + fwd * 0.2
            e = energy * 0.35
        else:                                       # table / floor lamps: bulb in the upper half of the shade
            pos = Vector((c.x, c.y, mn.z + 0.75 * (mx.z - mn.z)))
            e = energy * 0.5
        ld = bpy.data.lights.new(name=f"lamp_light_{o.name}", type="POINT")
        ld.energy = e
        ld.color = kelvin_rgb
        ld.shadow_soft_size = 0.08
        lo = bpy.data.objects.new(ld.name, ld)
        lo.location = pos
        scene.collection.objects.link(lo)
        made.append(lo.name)
        if len(made) >= max_lamps:
            break
    return made


def apply_plan(scene, plan: dict):
    """Move / remove the objects of a rearrangement plan (object names are those of the layout json)."""
    applied, missing = 0, []
    for ch in plan["changes"]:
        o = scene.objects.get(ch["name"])
        if o is None:
            missing.append(ch["name"]); continue
        if ch["op"] in ("remove", "follow_remove"):
            o.hide_render = True
            o.hide_viewport = True
            applied += 1
            continue
        fx, fy = ch["from_xy"]; tx, ty = ch["to_xy"]
        pv = ch.get("pivot_xy") or ch["from_xy"]
        c = Vector((pv[0], pv[1], 0.0))
        # rigid motion: rotate about the pivot, then translate so that the pivot lands on pivot + (to - from) of the pivot owner
        dyaw = ch.get("dyaw", 0.0)
        R = Matrix.Rotation(dyaw, 4, "Z")
        if ch["op"] == "follow":
            # the child's to_xy already includes the parent's rotation; translation = to - R*(from - pivot) - pivot
            rf = R @ Vector((fx - pv[0], fy - pv[1], 0.0))
            d = Vector((tx - pv[0] - rf.x, ty - pv[1] - rf.y, ch.get("dz", 0.0)))
        else:
            d = Vector((tx - fx, ty - fy, ch.get("dz", 0.0)))
        M = Matrix.Translation(d) @ Matrix.Translation(c) @ R @ Matrix.Translation(-c) @ o.matrix_world
        o.matrix_world = M
        applied += 1
    return applied, missing


def add_id_pass(scene, tree, rl, out_dir: Path):
    """File output of the object-index pass (pass_index set by the importer; stage = 0)."""
    vl = scene.view_layers[0]
    vl.use_pass_object_index = True
    tree.update_tag() if hasattr(tree, "update_tag") else None
    sock = None
    for name in ("IndexOB", "Index Object", "Object Index", "IndexOB.001"):
        if name in rl.outputs:
            sock = rl.outputs[name]; break
    if sock is None:
        cands = [o for o in rl.outputs if "index" in o.name.lower() and ("ob" in o.name.lower())]
        if not cands:
            raise RuntimeError(f"no object-index output on the render layers node; outputs: {[o.name for o in rl.outputs]}")
        sock = cands[0]
    fo = tree.nodes.new("CompositorNodeOutputFile")
    if hasattr(fo, "directory"):
        fo.directory = str(out_dir)
        fo.file_name = "id_"
        fo.file_output_items.new("FLOAT", "id")
        tree.links.new(sock, fo.inputs["id"])
        fo.format.file_format = "OPEN_EXR_MULTILAYER"
    else:
        fo.base_path = str(out_dir)
        fo.file_slots[0].path = "id_"
        tree.links.new(sock, fo.inputs[0])
        fo.format.file_format = "OPEN_EXR"
        fo.format.color_mode = "RGB"
    fo.format.color_depth = "32"
    return fo


def fix_glass(scene, min_transmission=0.5):
    """Window glass imported from glTF (Principled transmission) blocks light in Cycles without caustics: route shadow and
    diffuse rays through a Transparent BSDF so that daylight enters, camera rays keep the glass look."""
    fixed = []
    for m in bpy.data.materials:
        if not m.use_nodes or m.node_tree is None:
            continue
        nt = m.node_tree
        if any(n.type == "LIGHT_PATH" and n.get("hssd_glass_fix") for n in nt.nodes):
            continue
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None) or \
            next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
        if out is None or not out.inputs["Surface"].is_linked:
            continue
        src = out.inputs["Surface"].links[0].from_node
        is_glass = False
        if src.type == "BSDF_PRINCIPLED":
            for key in ("Transmission Weight", "Transmission"):
                if key in src.inputs and not src.inputs[key].is_linked and src.inputs[key].default_value >= min_transmission:
                    is_glass = True
            if "Alpha" in src.inputs and not src.inputs["Alpha"].is_linked and src.inputs["Alpha"].default_value < 0.5:
                is_glass = True
        elif src.type in ("BSDF_GLASS", "BSDF_REFRACTION"):
            is_glass = True
        if not is_glass:
            continue
        lp = nt.nodes.new("ShaderNodeLightPath"); lp["hssd_glass_fix"] = True
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        math_max = nt.nodes.new("ShaderNodeMath"); math_max.operation = "MAXIMUM"
        nt.links.new(lp.outputs["Is Shadow Ray"], math_max.inputs[0])
        nt.links.new(lp.outputs["Is Diffuse Ray"], math_max.inputs[1])
        nt.links.new(math_max.outputs[0], mix.inputs["Fac"])
        nt.links.new(src.outputs[0], mix.inputs[1])
        nt.links.new(tr.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
        fixed.append(m.name)
    return fixed
