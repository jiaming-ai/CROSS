#!/usr/bin/env python3
"""Build a Blender scene (.blend) from an HSSD (hssd-hab) scene instance.

* stage glb + every object instance (Habitat Y-up -> Blender Z-up; rotation quaternion is [w, x, y, z];
  translation_origin is asset_local for these scenes)
* one mesh per template, instances share the mesh; every instance carries custom properties
  hssd_template / hssd_category / hssd_super / hssd_name / hssd_index / hssd_maxdim and a unique pass_index
* light portals at windows/doors, a Nishita sky world, no other lights (the generator adds lamps per preset)
* layout.json with per-instance world bounding boxes, stage bounds and room regions (Blender frame)
* optional Workbench top-down preview

    .venv-blender/bin/python scripts/sim/hssd_to_blend.py --root data/sim/assets/hssd --scene 106366323_174226647 \
        --out data/sim/assets/hssd_106366323.blend --preview
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glb_debasis  # noqa: E402

KTX_BIN = str(Path.home() / "opt/ktx/bin/ktx")


def ensure_importable(path: Path):
    """Convert KHR_texture_basisu textures to PNG in place (needs the ktx CLI); returns False if impossible."""
    js, _ = glb_debasis.read_glb(path)
    if "KHR_texture_basisu" not in (js.get("extensionsUsed") or []):
        return True
    try:
        glb_debasis.convert(path, KTX_BIN)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  cannot convert {path.name}: {e}", flush=True)
        return False

C_HAB2BL = Matrix.Rotation(math.radians(90.0), 4, "X")      # (x, y, z)_hab -> (x, -z, y)_blender
C_INV = C_HAB2BL.inverted()
LAMP_CATS = {"ceiling_lamp", "wall_lamp", "table_lamp", "floor_lamp", "lamp", "chandelier", "pendant_lamp"}
PORTAL_CATS = {"window", "door", "opening"}


def template_glb(root: Path, name: str) -> Path | None:
    """Render asset for a template name; a decomposed part (<hash>_part_N) uses its own (texture-converted) glb when
    present, otherwise the caller falls back to the whole object <hash>."""
    base = name.split("_part_")[0]
    if "_part_" in name:
        p = root / f"objects/decomposed/{base}/{name}.glb"
        return p if p.is_file() else None
    if base.startswith("xxxx"):
        cands = [root / f"objects/x/{base}.glb"]
    elif "-" in base and len(base) <= 8:
        cands = [root / f"objects/openings/{base}.glb"]
    else:
        cands = [root / f"objects/{base[0]}/{base}.glb", root / f"objects/x/{base}.glb"]
    for c in cands:
        if c.is_file():
            return c
    return None


def import_glb_as_single_mesh(path: Path, name: str):
    """Import a glb and merge all its meshes into one object with identity transform (vertices in Blender frame)."""
    if not ensure_importable(path):
        return None
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.gltf(filepath=str(path), import_shading="NORMALS")
    except RuntimeError as e:
        print(f"  import failed for {path.name}: {e}", flush=True)
        for o in [o for o in bpy.data.objects if o not in before]:
            bpy.data.objects.remove(o, do_unlink=True)
        return None
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    others = [o for o in new if o.type != "MESH"]
    if not meshes:
        for o in others:
            bpy.data.objects.remove(o, do_unlink=True)
        return None
    # bake the hierarchy into world transforms, then join
    for o in meshes:
        mw = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = mw
    main = meshes[0]
    if len(meshes) > 1:
        with bpy.context.temp_override(active_object=main, selected_editable_objects=meshes, selected_objects=meshes):
            bpy.ops.object.join()
    with bpy.context.temp_override(active_object=main, selected_editable_objects=[main], selected_objects=[main]):
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for o in others:
        bpy.data.objects.remove(o, do_unlink=True)
    main.name = name
    main.data.name = name
    return main


def world_bbox(o):
    pts = [o.matrix_world @ Vector(c) for c in o.bound_box]
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return mn, mx


def load_categories(root: Path):
    cat = {}
    p = root / "semantics/objects.csv"
    if not p.is_file():
        return cat
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            main = r.get("main_category") or ""
            wn = (r.get("wnsynsetkey") or "").split(".")[0]
            tags = (r.get("floorplanner-category-tags") or "").split(",")[0].strip().lower().replace(" ", "_").replace("&", "and")
            cat[r["id"]] = {"category": main or wn or tags or "unknown", "super": r.get("super_category") or "",
                            "name": r.get("name") or "", "dims": r.get("aligned.dims") or "", "tags": r.get("floorplanner-category-tags") or ""}
    return cat


def hab_to_bl_point(p):
    v = C_HAB2BL @ Vector((p[0], p[1], p[2], 1.0))
    return [v.x, v.y, v.z]


def make_world_sky(scene):
    w = bpy.data.worlds.new("hssd_world")
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    sky = nt.nodes.new("ShaderNodeTexSky")
    try:
        sky.sky_type = "MULTIPLE_SCATTERING"
    except TypeError:
        sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(45.0)
    sky.sun_rotation = 0.0
    sky.sun_intensity = 1.0
    bg.inputs["Strength"].default_value = 1.0
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    scene.world = w


def add_portal(coll, inst, center_xy_house):
    mn, mx = world_bbox(inst)
    ext = mx - mn
    axes = sorted(range(3), key=lambda k: ext[k])
    thin = axes[0]
    if thin == 2:                     # horizontal opening (skylight) - skip
        return None
    a, b = [k for k in range(3) if k != thin]
    size_a, size_b = ext[a], ext[b]
    if max(size_a, size_b) < 0.4:
        return None
    ld = bpy.data.lights.new(f"portal_{inst.name}", type="AREA")
    ld.shape = "RECTANGLE"
    ld.size = float(ext[0] if thin == 1 else ext[1])       # x-extent if the window lies in the xz plane, else y-extent
    ld.size_y = float(ext[2])
    ld.cycles.is_portal = True
    lo = bpy.data.objects.new(ld.name, ld)
    c = (mn + mx) / 2
    lo.location = c
    normal = Vector((1.0, 0.0, 0.0)) if thin == 0 else Vector((0.0, 1.0, 0.0))
    # face the interior: towards the house centre
    to_center = Vector((center_xy_house[0] - c.x, center_xy_house[1] - c.y, 0.0))
    if normal.dot(to_center) < 0:
        normal = -normal
    lo.rotation_mode = "QUATERNION"
    lo.rotation_quaternion = normal.to_track_quat("-Z", "Y")     # area lights emit along -Z
    coll.objects.link(lo)
    return lo.name


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--preview-res", type=int, default=2400)
    ap.add_argument("--max-objects", type=int, default=0)
    ap.add_argument("--preview-only", action="store_true", help="open the existing --out blend and only render the previews")
    args = ap.parse_args(argv)
    root, out = Path(args.root), Path(args.out)
    t0 = time.time()
    if args.preview_only:
        bpy.ops.wm.open_mainfile(filepath=str(out))
        scene = bpy.context.scene
        lay = json.loads(Path(str(out) + ".layout.json").read_text())
        smn, smx = Vector(lay["stage_bbox"][0]), Vector(lay["stage_bbox"][1])
        cx, cy = (smn.x + smx.x) / 2, (smn.y + smx.y) / 2
        render_topdown(scene, out, smn, smx, cx, cy, args.preview_res)
        return

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    for cname in ("Stage", "Objects", "Templates", "Portals"):
        coll = bpy.data.collections.new(cname)
        scene.collection.children.link(coll)
    C = {c.name: c for c in bpy.data.collections}
    # import into the Templates collection
    lc = next(l for l in bpy.context.view_layer.layer_collection.children if l.name == "Templates")
    bpy.context.view_layer.active_layer_collection = lc

    sj = json.loads((root / f"scenes/{args.scene}.scene_instance.json").read_text())
    cats = load_categories(root)
    stage_name = sj["stage_instance"]["template_name"].split("/")[-1]

    # ---- stage
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(root / f"stages/{stage_name}.glb"), import_shading="NORMALS")
    stage_objs = [o for o in bpy.data.objects if o not in before]
    smn, smx = Vector((1e9,) * 3), Vector((-1e9,) * 3)
    stage_info = []
    for o in stage_objs:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        C["Stage"].objects.link(o)
        if o.type == "MESH":
            mw = o.matrix_world.copy(); o.parent = None; o.matrix_world = mw
            with bpy.context.temp_override(active_object=o, selected_editable_objects=[o], selected_objects=[o]):
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            mn, mx = world_bbox(o)
            smn = Vector((min(smn[k], mn[k]) for k in range(3))); smx = Vector((max(smx[k], mx[k]) for k in range(3)))
            o.name = "stage_" + o.name
            stage_info.append({"name": o.name, "verts": len(o.data.vertices), "materials": [m.name for m in o.data.materials if m],
                               "bbox": [list(mn), list(mx)]})
    for o in stage_objs:
        if o.type != "MESH":
            bpy.data.objects.remove(o, do_unlink=True)
    print(f"stage: {len(stage_info)} meshes, bbox {tuple(round(v, 2) for v in smn)} - {tuple(round(v, 2) for v in smx)}", flush=True)

    # ---- objects
    insts = sj["object_instances"]
    if args.max_objects:
        insts = insts[: args.max_objects]
    templates = {}
    layout = []
    missing = []
    seen_parts = set()
    n_parts_merged = 0
    for i, inst in enumerate(insts):
        tn = inst["template_name"]
        base = tn.split("_part_")[0]
        if "_part_" in tn and template_glb(root, tn) is None:
            # part geometry unavailable: place the whole source object once per distinct transform
            key = (base, tuple(round(v, 3) for v in inst["translation"]), tuple(round(v, 3) for v in inst.get("rotation", [1, 0, 0, 0])))
            if key in seen_parts:
                n_parts_merged += 1
                continue
            seen_parts.add(key)
            tn = base
        if tn not in templates:
            p = template_glb(root, tn)
            if p is None:
                missing.append(tn); templates[tn] = None
            else:
                templates[tn] = import_glb_as_single_mesh(p, f"tmpl_{tn}")
        tmpl = templates[tn]
        if tmpl is None:
            continue
        info = cats.get(base, {"category": "opening" if ("-" in tn and len(tn) <= 8) else "unknown", "super": "", "name": "", "dims": ""})
        cat = info["category"]
        t = inst["translation"]
        q = inst.get("rotation", [1, 0, 0, 0])
        s = inst.get("non_uniform_scale", [1, 1, 1])
        Mh = Matrix.Translation(Vector(t)) @ Quaternion((q[0], q[1], q[2], q[3])).to_matrix().to_4x4() @ Matrix.Diagonal((s[0], s[1], s[2], 1.0))
        o = bpy.data.objects.new(f"hssd_{cat}", tmpl.data)
        o.matrix_world = C_HAB2BL @ Mh @ C_INV
        o.pass_index = i + 1
        o["hssd_template"] = tn
        o["hssd_category"] = cat
        o["hssd_super"] = info["super"]
        o["hssd_name"] = info["name"]
        o["hssd_index"] = i
        try:
            o["hssd_maxdim"] = max(float(x) for x in info["dims"].split(","))
        except Exception:  # noqa: BLE001
            o["hssd_maxdim"] = -1.0
        o["hssd_lamp"] = cat in LAMP_CATS
        C["Objects"].objects.link(o)
        mn, mx = world_bbox(o)
        layout.append({"name": o.name, "index": i, "template": tn, "category": cat, "super": info["super"], "label": info["name"][:60],
                       "bbox_min": [round(v, 4) for v in mn], "bbox_max": [round(v, 4) for v in mx],
                       "matrix_world": [list(r) for r in o.matrix_world], "maxdim": o["hssd_maxdim"]})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(insts)} instances, {len(templates)} templates, {time.time() - t0:.0f}s", flush=True)
    # hide the template objects from render/view (they sit at the origin)
    for tmpl in templates.values():
        if tmpl is not None:
            tmpl.hide_render = True
            tmpl.hide_viewport = True
    lc.exclude = True

    # ---- portals + world
    make_world_sky(scene)
    cx, cy = (smn.x + smx.x) / 2, (smn.y + smx.y) / 2
    portals = []
    for o in C["Objects"].objects:
        if o.get("hssd_category") in PORTAL_CATS:
            n = add_portal(C["Portals"], o, (cx, cy))
            if n:
                portals.append(n)
    print(f"{len(layout)} instances placed ({n_parts_merged} decomposed parts merged into whole objects), {len(templates)} templates, {len(missing)} missing: {missing[:10]}, {len(portals)} portals; {time.time() - t0:.0f}s", flush=True)

    # ---- regions
    regions = []
    sp = root / f"semantics/scenes/{args.scene}.semantic_config.json"
    if sp.is_file():
        try:
            sd = json.loads(sp.read_text())
            for r in sd.get("region_annotations", []):
                poly = [hab_to_bl_point(p) for p in r.get("poly_loop", [])]
                regions.append({"name": r.get("name"), "poly": poly, "floor_height": r.get("floor_height"), "extrusion_height": r.get("extrusion_height")})
        except Exception as e:  # noqa: BLE001
            print("region parse failed:", e)

    # ---- render settings for later use
    scene.render.engine = "CYCLES"
    scene.view_settings.view_transform = "AgX"
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out), compress=True)
    lay = {"scene": args.scene, "stage_bbox": [list(smn), list(smx)], "stage": stage_info, "objects": layout, "regions": regions,
           "missing_templates": missing, "portals": portals}
    Path(str(out) + ".layout.json").write_text(json.dumps(lay, indent=0))
    print(f"saved {out} ({out.stat().st_size / 1e6:.0f} MB) + layout json; {time.time() - t0:.0f}s", flush=True)

    if args.preview:
        render_topdown(scene, out, smn, smx, cx, cy, args.preview_res)


def render_topdown(scene, out, smn, smx, cx, cy, res):
    if True:
        # Workbench top-down orthographic view; the camera plane at 2.3 m clips ceilings and roofs
        cam_data = bpy.data.cameras.new("topdown")
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = float(max(smx.x - smn.x, smx.y - smn.y) * 1.04)
        cam_data.clip_start = 0.01
        cam_data.clip_end = 50.0
        cam = bpy.data.objects.new("topdown", cam_data)
        scene.collection.objects.link(cam)
        cam.location = (cx, cy, 2.3)
        cam.rotation_euler = (0.0, 0.0, 0.0)
        scene.camera = cam
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "TEXTURE"
        scene.display.shading.show_shadows = False
        scene.display.shading.show_cavity = True
        scene.render.resolution_x = scene.render.resolution_y = res
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(out) + ".topdown.png"
        bpy.ops.render.render(write_still=True)
        # second view at 5.5 m to include tall objects / show the roof-less structure
        cam.location = (cx, cy, 5.5)
        scene.render.filepath = str(out) + ".topdown_5m.png"
        bpy.ops.render.render(write_still=True)
        print("preview written", flush=True)


if __name__ == "__main__":
    main()
