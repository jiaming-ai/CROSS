#!/usr/bin/env python3
"""SimChange: render controlled multi-traversal stereo sequences with Blender (bpy).

One base trajectory per scene is rendered as the *map* traversal.  Every *query*
traversal changes exactly one factor (illumination, object movement/removal,
background materials, viewpoint offset / yaw / height, traversal direction) at a
quantified level.  Output per traversal:

    left/000000.png right/000000.png depth/000000.npy (float16, metres, left cam)
    poses_left.txt   (N x 16, camera-to-world, OpenCV camera convention, Blender world)
    calib.json       (K, width, height, baseline, T_right_in_left, fps, scene, variant)
    meta.json        (variant parameters, moved/removed objects, light factors)

Run with the bpy environment:
    .venv-blender/bin/python scripts/sim/gen_simchange.py --scene classroom --variants map,light_0.25 --out data/sim/classroom
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ARGS_GPU_INDEX = -1
sys.path.insert(0, str(Path(__file__).resolve().parent))

SCENES = {
    "classroom": {
        "blend": ROOT / "data/sim/assets/classroom/classroom.blend",
        "engine": "CYCLES", "samples": 24,
        # U-shaped path through two aisles; xy waypoints (m), camera height (m)
        "waypoints": [(-1.95, -3.6), (-1.95, 0.9), (0.85, 0.9), (0.85, -4.0)],
        "height": 1.2,
        "radius": 0.3,
        "exclude_collections": ["volumeLight"],   # volumetric god-rays: 3x render cost, no geometric value
        "bounds": (-3.0, 3.3, -4.6, 3.2),
        "movable": ["leatherChair", "dustBin", "suitcase", "coatStand", "crinkledPaper", "beigeBook",
                    "leatherBook", "zapBook", "schoolDesk"],
        "movable_exclude": [],
        "walls": ["wall", "sol", "woodBase", "woodBaseboard"],
        "night_keep": ["coridor_ceilingLight", "blackBoard_light"],
        # lighting rig: the sun shines through the windows on the +x wall; daylight also enters via an
        # emissive "portal" plane; interior lamps are emissive ceiling-lamp meshes + two small lights
        "sun_light": "sun", "sun_energy": 1.0, "window_azimuth_deg": 0.0, "window_center": (3.5, -1.0, 1.5),
        "portal_materials": ["dayLight_portal"], "portal_strength": 20.0,
        "lamp_lights": ["coridor_ceilingLight", "blackBoard_light"],
        "lamp_materials": ["ceillingLamp_light", "blackBoardLight"],
        "fill_lights": ["exterior_fillLight"],
    },
    "lonemonk": {
        # Blender demo "Lone Monk" (Carlo Bergonzini, CC-BY): a monastery cloister, 40 x 40 m footprint.
        # Long loop: west arcade -> north arcade -> east arcade -> along the church wall -> through the
        # courtyard around the well and back into the west arcade (~120 m, ~1200 frames at 0.1 m).
        "blend": ROOT / "data/sim/assets/lone_monk.blend",
        "engine": "CYCLES", "samples": 16,
        "waypoints": [(10.4, 13.0), (10.4, 29.5), (29.4, 29.5), (29.4, 13.0), (24.0, 13.0), (14.0, 13.5),
                      (16.5, 17.5), (23.0, 22.5), (25.5, 26.0), (14.5, 26.0), (12.5, 20.0), (10.4, 15.0)],
        "height": 1.4,
        "radius": 0.4,
        "bounds": (9.5, 30.5, 11.5, 30.5),
        "movable": ["chair_main", "GothicCommode", "gotchi_commode_box", "bookshelf", "blk_bench", "flower_low", "bookOPEN", "chair"],
        "movable_exclude": [],
        "walls": ["blk_wall", "blk_floor", "arch", "cloumn_belt", "roof", "terracotta moulding"],
        "night_keep": [],
        # lighting rig: the sun/sky is a Nishita sky texture in the world; at night/evening point lights are
        # created at the wall torches
        "sky_nishita": True, "sun_light": None, "sun_energy": 1.0, "window_azimuth_deg": 0.0,
        "torch_objects": "torch", "torch_energy": 400.0,
        "portal_materials": [], "portal_strength": 0.0, "lamp_lights": [], "lamp_materials": [],
    },
    "hssd_103997940": {
        # HSSD scene 103997940_171031257 (Habitat Synthetic Scenes Dataset, CC BY-NC 4.0): a 57 x 36 m single-floor house
        # with 36 rooms and 639 separately placed objects.  Rearrangement variants use scripts/sim/rearrange_plan.py.
        "blend": ROOT / "data/sim/assets/hssd_103997940.blend",
        "layout": ROOT / "data/sim/assets/hssd_103997940.blend.layout.json",
        "engine": "CYCLES", "samples": 16,
        "waypoints": [],                       # set by --waypoints / filled in after the layout inspection
        "height": 1.3,
        "radius": 0.35,
        "bounds": (-51.5, 14.0, -11.2, 30.1),
        "movable": [], "movable_exclude": [],
        "walls_prefix": "stage_", "walls": [],
        "night_keep": [],
        "sky_nishita": True, "sun_light": None, "sun_energy": 1.0, "window_azimuth_deg": 0.0,
        "lamp_prop": "hssd_lamp", "torch_objects": None, "torch_energy": 300.0,
        "exposure": {"morning": 0.8, "noon": 0.3, "afternoon": 0.6, "evening": 1.5, "night": 2.0, "overcast": 1.0},
        "portal_materials": [], "portal_strength": 0.0, "lamp_lights": [], "lamp_materials": [],
    },
    "hssd_104348010": {
        # HSSD scene 104348010_171512832: a 41 x 24 m single-floor house (kitchen, dining, two living rooms, five bedrooms,
        # office, bathrooms, garage, furnished terrace) with 708 separately placed objects.
        "blend": ROOT / "data/sim/assets/hssd_104348010.blend",
        "layout": ROOT / "data/sim/assets/hssd_104348010.blend.layout.json",
        "engine": "CYCLES", "samples": 16,
        # ~195 m mapping loop with a same-direction loop closure: garage -> west hallway -> kitchen (around the island) ->
        # terrace -> dining -> living room -> entry -> east hallway -> utility room -> bedroom 2 -> office -> bedroom 1 ->
        # hallway -> living room -> and again hallway end -> kitchen -> terrace (the last ~30 m repeat stations ~90-380)
        "waypoints": [(-17, 3), (-15, 6.6), (-12, 6.7), (-11.5, 8.6), (-11.5, 12), (-7.5, 12.2), (-8.5, 14.3), (-6, 16.5), (-3.5, 14.5),
                      (-3.3, 12.5), (-3.2, 10.3), (-6, 6.2), (-11, 4.5), (-11, 1), (-6, 0.3), (-1.3, 0.5), (-1.5, 4), (0.5, 6.4),
                      (9, 6.4), (11.5, 6.9), (7, 7.6), (8.6, 11.5), (7, 7.6), (4, 7.6), (4, 11), (4, 7.6), (1.5, 5.5), (2.2, 2),
                      (1.5, 5.5), (0.3, 6.4), (-6, 6.3), (-12, 6.7), (-11.5, 8.6), (-11.5, 12), (-7.5, 12.2), (-8.5, 14.3), (-6, 16.5)],
        "height": 1.3,
        "radius": 0.35,
        "bounds": (-23.4, 17.9, -6.2, 20.5),
        "hide_categories": ["door"], "fix_glass": True, "exposure_base": 3.0, "restrict_to_regions": True, "max_turn_deg_per_frame": 10.0,
        "auto_exposure": {"regions": ["terrace", "outdoor"], "offset": {"default": -2.5, "overcast": -2.0, "evening": -0.5, "night": 0.0}},
        "movable": [], "movable_exclude": [],
        "walls_prefix": "stage_", "walls": [],
        "night_keep": [],
        "sky_nishita": True, "sun_light": None, "sun_energy": 1.0, "window_azimuth_deg": 0.0,
        "lamp_prop": "hssd_lamp", "torch_objects": None, "torch_energy": 40.0,
        # exposure offsets are relative to exposure_base (indoor daylight)
        "exposure": {"morning": 0.3, "noon": 0.0, "afternoon": 0.2, "evening": -2.0, "night": 0.0, "overcast": 0.5},
        "portal_materials": [], "portal_strength": 0.0, "lamp_lights": [], "lamp_materials": [],
    },
    "hssd_103997718": {
        # HSSD scene 103997718_171030855: a 49 x 44 m waterfront restaurant / bar (dining hall, booths, bar, lounge, games
        # room, kitchen, toilets, terrace) with 1410 separately placed objects (703 distinct models).
        "blend": ROOT / "data/sim/assets/hssd_103997718.blend",
        "layout": ROOT / "data/sim/assets/hssd_103997718.blend.layout.json",
        "engine": "CYCLES", "samples": 16,
        # ~210 m mapping loop with a same-direction loop closure: dining hall east aisle -> games corner -> service corridor ->
        # lounge / communal tables -> west aisle -> terrace door -> terrace (umbrellas) and back -> west aisle -> corridor ->
        # kitchen -> dining hall west aisle -> north side -> start -> again east aisle to the corridor (repeat of the first ~25 m)
        "waypoints": [(-6.81, 9.28), (-6.81, 0.98), (-5.51, -1.22), (-9.01, -3.02), (-13.01, -3.02), (-17.01, -3.32), (-24.01, -3.32),
                      (-26.41, -3.02), (-27.81, 2.63), (-26.96, 8.48), (-27.81, 10.98), (-27.01, 13.48), (-23.01, 15.48), (-19.01, 13.98),
                      (-23.11, 13.48), (-27.01, 12.48), (-27.81, 10.98), (-26.96, 8.48), (-27.81, 2.63), (-26.41, -3.02), (-20.01, -3.27),
                      (-17.51, -5.12), (-16.31, -9.22), (-12.36, -9.97), (-7.21, -9.82), (-6.01, -5.02), (-9.01, -3.02), (-10.81, -1.22),
                      (-11.41, 3.08), (-12.01, 9.78), (-9.01, 9.78), (-6.81, 9.28), (-6.81, 0.98), (-5.51, -1.22), (-9.01, -3.02)],
        "height": 1.3,
        "radius": 0.3,
        "bounds": (-46.5, 2.5, -13.3, 30.9),
        "hide_categories": ["door"], "fix_glass": True, "exposure_base": 2.7, "restrict_to_regions": True, "max_turn_deg_per_frame": 10.0,
        "auto_exposure": {"regions": ["terrace", "outdoor"], "offset": {"default": -2.5, "overcast": -2.0, "evening": -0.5, "night": 0.0}},
        "movable": [], "movable_exclude": [],
        "walls_prefix": "stage_", "walls": [],
        "night_keep": [],
        "sky_nishita": True, "sun_light": None, "sun_energy": 1.0, "window_azimuth_deg": 0.0,
        "lamp_prop": "hssd_lamp", "torch_objects": None, "torch_energy": 20.0,
        "exposure": {"morning": 0.0, "noon": -0.3, "afternoon": -0.2, "evening": -2.5, "night": -1.0, "overcast": 0.2},
        "portal_materials": [], "portal_strength": 0.0, "lamp_lights": [], "lamp_materials": [],
    },
    "archiviz": {
        "blend": ROOT / "data/sim/assets/archiviz.blend",
        "engine": "BLENDER_EEVEE", "samples": 16, "unhide_all": True,
        "waypoints": [(0.2, -5.2), (0.2, -1.3), (3.2, -1.3), (5.8, -1.6), (5.8, -3.7), (2.5, -3.7)],
        "height": 1.3,
        "bounds": (-1.4, 8.0, -7.2, -0.2),
        "movable": ["Chair", "Pear", "Talir", "Tankard", "Thomas_Konvice", "Table-Rustik", "TV"],
        "movable_exclude": ["Chair_Thonet"],
        "walls": ["Walls", "FloorBoard", "Carpet", "Plane"],
        "night_keep": ["Hall", "WC", "Bathroom", "Area"],
        # lighting rig: a strong spot ("Sun") outside the windows plays the sun; interior spots are lamps;
        # the HDRI world provides sky light
        "sun_light": "Sun", "sun_energy": 6000.0, "sun_distance": 12.0, "window_center": (4.0, -6.0, 1.5),
        "lamp_gain_night": 40.0, "lamp_gain_evening": 12.0,
        "window_azimuth_deg": -30.0,
        "portal_materials": [], "portal_strength": 0.0,
        "lamp_lights": ["Hall", "WC", "Bathroom", "Area"],
        "lamp_materials": [],
        "fill_lights": [],
    },
}

# ----------------------------------------------------------------------------- #
# lighting presets: real-world illumination changes are modelled by *different light sources*, not
# by dimming: sun elevation/azimuth (relative to the window direction) and colour temperature,
# sky/portal brightness and colour, and interior lamps switched on or off.
LIGHT_PRESETS = {
    #            elev  azim  sun_K  sun_gain  sky_gain sky_K  lamps
    "morning":   dict(elev=12.0, azim=-35.0, sun_K=4200, sun_gain=2.5, sky_gain=0.6, sky_K=7500, lamps=False),
    "noon":      dict(elev=62.0, azim=0.0,   sun_K=6000, sun_gain=4.0, sky_gain=1.2, sky_K=6500, lamps=False),
    "afternoon": dict(elev=32.0, azim=40.0,  sun_K=5200, sun_gain=3.0, sky_gain=1.0, sky_K=6500, lamps=False),
    "evening":   dict(elev=4.0,  azim=55.0,  sun_K=2500, sun_gain=1.5, sky_gain=0.25, sky_K=4000, lamps=True),
    "night":     dict(elev=None, azim=0.0,   sun_K=6500, sun_gain=0.0, sky_gain=0.02, sky_K=9000, lamps=True, lamp_gain=2.0),
    "overcast":  dict(elev=None, azim=0.0,   sun_K=6500, sun_gain=0.0, sky_gain=2.0, sky_K=7000, lamps=False),
}


def blackbody_rgb(kelvin: float):
    """Approximate linear RGB of a blackbody (Tanner Helland fit, normalised to max 1)."""
    t = max(1000.0, min(40000.0, kelvin)) / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * ((t - 60) ** -0.1332047592)
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * ((t - 60) ** -0.0755148492)
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    c = np.clip(np.array([r, g, b]) / 255.0, 0, 1)
    c = c ** 2.2                                   # sRGB-ish -> linear
    return tuple(float(v) for v in c / max(c.max(), 1e-6))


def sun_direction(cfg, elev_deg, azim_deg):
    """Direction the sun light *shines along* (world), entering through the windows."""
    az = math.radians(cfg.get("window_azimuth_deg", 0.0) + azim_deg)
    el = math.radians(elev_deg)
    # the sun sits outside the window wall (window_azimuth points from the room towards the windows)
    outward = Vector((math.cos(az), math.sin(az), 0.0))
    sun_pos_dir = Vector((outward.x * math.cos(el), outward.y * math.cos(el), math.sin(el)))
    return -sun_pos_dir


def _torch_lights(scene, cfg, energy, kelvin=2200.0):
    """Point lights at the torch objects (outdoor scenes without lamp objects)."""
    made = []
    for o in list(scene.objects):
        if o.type == "MESH" and o.name.split(".")[0] == cfg["torch_objects"]:
            ld = bpy.data.lights.new(name=f"torch_light_{o.name}", type="POINT")
            ld.energy = energy
            ld.color = blackbody_rgb(kelvin)
            ld.shadow_soft_size = 0.15
            lo = bpy.data.objects.new(ld.name, ld)
            lo.location = o.matrix_world.translation + Vector((0, 0, 0.3))
            scene.collection.objects.link(lo)
            made.append(lo.name)
    return made


def apply_lighting(scene, cfg, preset_name, meta):
    p = LIGHT_PRESETS[preset_name]
    meta["lighting"] = {"preset": preset_name, **{k: v for k, v in p.items()}}
    if cfg.get("sky_nishita"):
        # outdoor scene lit by a physical sky: move the sun in the sky texture; the camera auto-exposure
        # of a real rig is mimicked by the view-transform exposure so that dusk/night are dark but not black
        w = scene.world
        sky = next((n for n in w.node_tree.nodes if n.type == "TEX_SKY"), None) if w and w.use_nodes else None
        expo = {"morning": 0.6, "noon": 0.0, "afternoon": 0.3, "evening": 1.8, "night": 3.0, "overcast": 1.0}[preset_name]
        expo = cfg.get("exposure", {}).get(preset_name, expo)
        if sky is not None:
            try:                                     # Blender 5: Nishita = single/multiple scattering
                sky.sky_type = "MULTIPLE_SCATTERING"
            except TypeError:
                sky.sky_type = "NISHITA"
            if p["elev"] is None:                      # night / overcast: no sun disc
                sky.sun_elevation = math.radians(-12.0 if preset_name == "night" else 35.0)
                sky.sun_intensity = 0.0
                if preset_name == "overcast":
                    for attr in ("dust_density", "aerosol_density"):
                        if hasattr(sky, attr):
                            setattr(sky, attr, 10.0)
            else:
                sky.sun_elevation = math.radians(p["elev"])
                sky.sun_intensity = p["sun_gain"] / 2.5
            sky.sun_rotation = math.radians(p["azim"] + cfg.get("window_azimuth_deg", 0.0))
            _scale_world(scene, {"night": 0.2, "overcast": 3.0}.get(preset_name, 1.0))
        scene.view_settings.exposure = float(getattr(scene.view_settings, "exposure", 0.0)) + expo
        meta["lighting"]["exposure"] = expo
        if p["lamps"]:
            if cfg.get("lamp_prop"):
                import hssd_scene
                meta["lamp_lights"] = hssd_scene.lamp_lights(scene, cfg.get("torch_energy", 300.0) * p.get("lamp_gain", 1.0), blackbody_rgb(2700.0))
                meta["n_lamp_lights"] = len(meta["lamp_lights"])
            else:
                meta["torch_lights"] = _torch_lights(scene, cfg, cfg.get("torch_energy", 400.0) * p.get("lamp_gain", 1.0))
        return
    sun = scene.objects.get(cfg.get("sun_light"))
    if sun is not None:
        if p["elev"] is None:
            sun.data.energy = 0.0
        else:
            d = sun_direction(cfg, p["elev"], p["azim"])
            sun.rotation_mode = "QUATERNION"
            sun.rotation_quaternion = d.to_track_quat("-Z", "Y")
            if sun.data.type == "SPOT":                    # apartment: move the spot to where the sun would be
                c = Vector(cfg["window_center"])
                sun.location = c - d * cfg.get("sun_distance", 14.0)
                sun.data.spot_size = math.radians(70)
                sun.data.energy = cfg["sun_energy"] * p["sun_gain"] * 0.5
            else:
                sun.data.energy = cfg["sun_energy"] * p["sun_gain"]
            sun.data.color = blackbody_rgb(p["sun_K"])
    # daylight portals (emissive planes in the windows) follow the sky
    for m in bpy.data.materials:
        if m.name in cfg.get("portal_materials", []) and m.use_nodes:
            for n in m.node_tree.nodes:
                if n.type == "EMISSION":
                    n.inputs["Strength"].default_value = cfg["portal_strength"] * p["sky_gain"]
                    n.inputs["Color"].default_value = (*blackbody_rgb(p["sky_K"]), 1.0)
    for name in cfg.get("fill_lights", []):
        o = scene.objects.get(name)
        if o is not None:
            o.data.energy *= p["sky_gain"]
            o.data.color = blackbody_rgb(p["sky_K"])
    _scale_world(scene, p["sky_gain"] / max(cfg.get("world_base", 1.0), 1e-6) if cfg.get("world_base") else p["sky_gain"])
    # interior lamps
    gain = p.get("lamp_gain", 1.0) if p["lamps"] else 0.0
    if p["lamps"] and cfg.get(f"lamp_gain_{preset_name}") is not None:
        gain = cfg[f"lamp_gain_{preset_name}"]
    for name in cfg.get("lamp_lights", []):
        o = scene.objects.get(name)
        if o is not None:
            o.data.energy *= gain
    for m in bpy.data.materials:
        if m.name in cfg.get("lamp_materials", []) and m.use_nodes:
            for n in m.node_tree.nodes:
                if n.type == "EMISSION":
                    n.inputs["Strength"].default_value *= gain


# ----------------------------------------------------------------------------- #
# variants: name -> dict(kind=..., params)
# ----------------------------------------------------------------------------- #
def variant_spec(name: str) -> dict:
    if name == "map":
        return {"kind": "map"}
    if "+" in name:                                    # composite: merge the single-factor specs
        merged = {"kind": "multi", "parts": name.split("+")}
        for part in name.split("+"):
            sp = variant_spec(part)
            if sp["kind"] == "lighting":
                merged["preset"] = sp["preset"]
            elif sp["kind"] == "move":
                merged.update({"frac": sp["frac"], "remove": sp["remove"]})
            elif sp["kind"] == "background":
                merged["background"] = True
            elif sp["kind"] == "path":
                merged.update({k: v for k, v in sp.items() if k != "kind"})
            elif sp["kind"] == "rearrange":
                merged.update({"level": sp["level"], "rseed": sp["rseed"], "rops": sp.get("rops")})
            else:
                raise ValueError(f"cannot combine {part}")
        return merged
    if name.startswith("rearr_"):
        # rearr_50 | rearr_50_s2 | rearr_50_remove (op override) | rearr_50_s1_relocate
        toks = name.split("_")[1:]
        lvl = float(toks[0]) / 100.0
        seed, ops = 0, None
        for t in toks[1:]:
            if t.startswith("s") and t[1:].isdigit():
                seed = int(t[1:])
            elif t in ("remove", "relocate", "jitter", "swap"):
                ops = f"{t}:1.0"
        return {"kind": "rearrange", "level": lvl, "rseed": seed, "rops": ops}
    kind, _, lvl = name.partition("_")
    if kind == "light":
        if lvl in LIGHT_PRESETS:
            return {"kind": "lighting", "preset": lvl}
        return {"kind": "light", "factor": float(lvl)}
    if kind == "night":
        return {"kind": "lighting", "preset": "night"}
    if kind == "move":
        return {"kind": "move", "frac": float(lvl) / 100.0, "remove": False}
    if kind == "remove":
        return {"kind": "move", "frac": float(lvl) / 100.0, "remove": True}
    if kind == "background":
        return {"kind": "background"}
    if kind == "offset":
        return {"kind": "path", "lateral": float(lvl)}
    if kind == "height":
        return {"kind": "path", "dz": float(lvl)}
    if kind == "yaw":
        return {"kind": "path", "yaw": math.radians(float(lvl))}
    if kind == "reverse":
        return {"kind": "path", "reverse": True}
    if kind == "half":
        return {"kind": "path", "half": True}
    if kind == "combo":
        return {"kind": "combo"}
    raise ValueError(name)


# ----------------------------------------------------------------------------- #
def catmull_rom(points, step=0.1):
    """Resample a polyline through the waypoints with rounded corners (Catmull-Rom)."""
    P = np.asarray(points, dtype=np.float64)
    if len(P) < 2:
        raise ValueError("need >= 2 waypoints")
    Pe = np.concatenate([P[:1], P, P[-1:]])
    out = []
    for i in range(1, len(Pe) - 2):
        p0, p1, p2, p3 = Pe[i - 1], Pe[i], Pe[i + 1], Pe[i + 2]
        n = max(2, int(np.linalg.norm(p2 - p1) / 0.02))
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-1])
    dense = np.asarray(out)
    # arc-length resampling
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    targets = np.arange(0, s[-1], step)
    xy = np.stack([np.interp(targets, s, dense[:, k]) for k in range(2)], 1)
    # smoothed tangent
    d = np.gradient(xy, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    return xy, d


def plan_path(waypoints, occ, radius=0.35, step=0.1, allowed=None):
    """Clearance-aware shortest path through the occupancy grid between successive waypoints.

    Cells closer than `radius` to an obstacle are blocked; the remaining cells are
    weighted by 1/clearance so that the path prefers the middle of free space.  The
    result is resampled at `step` metres and lightly smoothed.
    """
    import heapq
    import cv2
    grid, x0, y0, res = occ["grid"], float(occ["x0"]), float(occ["y0"]), float(occ["res"])
    dist = cv2.distanceTransform((grid == 0).astype(np.uint8), cv2.DIST_L2, 5) * res
    H, W = grid.shape
    free = dist >= radius
    if allowed is not None:
        free &= allowed.astype(bool)
    cost = 1.0 + 2.0 / np.maximum(dist, 0.05)

    def to_cell(p):
        return int(round((p[1] - y0) / res)), int(round((p[0] - x0) / res))

    def nearest_free(c):
        iy, ix = c
        if free[iy, ix]:
            return c
        ys, xs = np.nonzero(free)
        k = np.argmin((ys - iy) ** 2 + (xs - ix) ** 2)
        return int(ys[k]), int(xs[k])

    def dijkstra(a, b):
        D = np.full((H, W), np.inf)
        prev = {}
        D[a] = 0.0
        pq = [(0.0, a)]
        nb = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == b:
                break
            if d > D[u]:
                continue
            for dy, dx, w in nb:
                v = (u[0] + dy, u[1] + dx)
                if not (0 <= v[0] < H and 0 <= v[1] < W) or not free[v]:
                    continue
                nd = d + w * cost[v]
                if nd < D[v]:
                    D[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if b not in prev and a != b:
            raise RuntimeError(f"no path between {a} and {b}")
        path = [b]
        while path[-1] != a:
            path.append(prev[path[-1]])
        return path[::-1]

    cells = [nearest_free(to_cell(p)) for p in waypoints]
    full = []
    for a, b in zip(cells[:-1], cells[1:]):
        seg = dijkstra(a, b)
        full.extend(seg if not full else seg[1:])
    pts = np.array([[x0 + c[1] * res, y0 + c[0] * res] for c in full])
    # smooth (moving average over ~0.5 m) and resample by arc length
    k = 9
    pad = np.pad(pts, ((k // 2, k // 2), (0, 0)), mode="edge")
    sm = np.stack([np.convolve(pad[:, i], np.ones(k) / k, mode="valid") for i in range(2)], 1)
    seg = np.linalg.norm(np.diff(sm, axis=0), axis=1)
    sarc = np.concatenate([[0], np.cumsum(seg)])
    targets = np.arange(0, sarc[-1], step)
    xy = np.stack([np.interp(targets, sarc, sm[:, i]) for i in range(2)], 1)
    d = np.gradient(xy, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    return xy, d


def region_mask(cfg, occ, dilate_m=0.4):
    """Cells inside any annotated room / outdoor region of the layout (dilated so that doorways are included)."""
    import cv2
    lay = json.loads(Path(cfg["layout"]).read_text())
    grid, x0, y0, res = occ["grid"], float(occ["x0"]), float(occ["y0"]), float(occ["res"])
    m = np.zeros(grid.shape, np.uint8)
    for r in lay.get("regions", []):
        if not r.get("poly") or len(r["poly"]) < 3:
            continue
        pts = np.array([[(p[0] - x0) / res, (p[1] - y0) / res] for p in r["poly"]], np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], 1)
    k = int(round(dilate_m / res)) * 2 + 1
    return cv2.dilate(m, np.ones((k, k), np.uint8)) > 0


def build_path(cfg, spec, occ=None):
    if occ is not None:
        allowed = region_mask(cfg, occ) if cfg.get("restrict_to_regions") and cfg.get("layout") else None
        xy, d = plan_path(cfg["waypoints"], occ, radius=cfg.get("radius", 0.35), step=0.1, allowed=allowed)
    else:
        xy, d = catmull_rom(cfg["waypoints"], step=0.1)
    yaw_off = spec.get("yaw", 0.0)
    if spec.get("lateral"):
        normal = np.stack([-d[:, 1], d[:, 0]], 1)  # left normal
        xy = xy - normal * spec["lateral"]          # positive = shift to the right of travel
    if spec.get("reverse"):
        xy, d = xy[::-1].copy(), -d[::-1].copy()
    if spec.get("half"):
        n = len(xy)
        xy, d = xy[n // 2:], d[n // 2:]
    z = cfg["height"] + spec.get("dz", 0.0)
    heading = np.arctan2(d[:, 1], d[:, 0]) + yaw_off
    return xy, z, heading


def insert_turns(xy, heading, max_step_deg):
    """Kinematic plausibility: where the heading changes by more than max_step_deg between consecutive stations
    (dead-end reversals, sharp corners of the grid path), insert in-place rotation stations so that no frame turns
    faster than max_step_deg.  Returns xy, heading and orig_index (old station index, -1 for inserted stations)."""
    step = math.radians(max_step_deg)
    out_xy, out_h, orig = [xy[0]], [heading[0]], [0]
    for i in range(1, len(xy)):
        dh = (heading[i] - heading[i - 1] + math.pi) % (2 * math.pi) - math.pi
        n = int(math.ceil(abs(dh) / step)) - 1
        for j in range(1, n + 1):
            out_xy.append(xy[i - 1]); out_h.append(heading[i - 1] + dh * j / (n + 1)); orig.append(-1)
        out_xy.append(xy[i]); out_h.append(heading[i]); orig.append(i)
    return np.asarray(out_xy), np.asarray(out_h), np.asarray(orig)


def clearance(xy, occ):
    """Minimum distance (m) from every path point to an occupied cell."""
    import cv2
    grid, x0, y0, res = occ["grid"], float(occ["x0"]), float(occ["y0"]), float(occ["res"])
    free = (grid == 0).astype(np.uint8)
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * res
    ix = np.clip(((xy[:, 0] - x0) / res).astype(int), 0, grid.shape[1] - 1)
    iy = np.clip(((xy[:, 1] - y0) / res).astype(int), 0, grid.shape[0] - 1)
    return dist[iy, ix]


# ----------------------------------------------------------------------------- #
def camera_matrix_blender(x, y, z, heading):
    """Blender camera world matrix: looks along `heading` in the xy plane, level, +Z up."""
    fwd = Vector((math.cos(heading), math.sin(heading), 0.0))
    up = Vector((0.0, 0.0, 1.0))
    zc = -fwd                       # Blender camera looks along -Z
    xc = up.cross(zc)               # right
    yc = zc.cross(xc)               # up
    M = Matrix.Identity(4)
    for i in range(3):
        M[i][0], M[i][1], M[i][2] = xc[i], yc[i], zc[i]
    M[0][3], M[1][3], M[2][3] = x, y, z
    return M


def c2w_opencv(M: Matrix) -> np.ndarray:
    T = np.array(M, dtype=np.float64)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return T @ flip


def setup_rig(scene, width, height, fov_deg, baselines):
    names = ["simL"] + [f"simR_{b:.2f}" for b in baselines]
    for name in names:
        if name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    cams = {}
    for name in names:
        data = bpy.data.cameras.new(name)
        data.sensor_fit = "HORIZONTAL"
        data.lens_unit = "FOV"
        data.angle = math.radians(fov_deg)
        data.clip_start = 0.05
        data.clip_end = 100.0
        cam = bpy.data.objects.new(name, data)
        scene.collection.objects.link(cam)
        cams[name] = cam
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    fx = width / 2.0 / math.tan(math.radians(fov_deg) / 2.0)
    K = np.array([[fx, 0, width / 2.0], [0, fx, height / 2.0], [0, 0, 1.0]])
    return cams, K


def setup_render(scene, cfg, samples, depth_dir):
    scene.render.engine = cfg["engine"]
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.use_motion_blur = False
    scene.render.use_compositing = True
    scene.render.use_persistent_data = True   # keep BVH / shaders between frames
    if cfg["engine"] == "CYCLES":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        cuda_devs = [d for d in prefs.devices if d.type == "CUDA"]
        for d in prefs.devices:
            d.use = False
        for i, d in enumerate(cuda_devs):
            d.use = (ARGS_GPU_INDEX < 0) or (i == ARGS_GPU_INDEX)
        scene.cycles.device = "GPU"
        print(f"[cycles] CUDA devices: {[d.name for d in cuda_devs]} enabled: {[d.name for d in cuda_devs if d.use]}", flush=True)
        if not any(d.use for d in cuda_devs):
            raise RuntimeError("no CUDA device enabled for Cycles")
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.max_bounces = 6
    else:
        scene.eevee.taa_render_samples = samples
    # depth pass via compositor file output (left camera only)
    vl = scene.view_layers[0]
    vl.use_pass_z = True
    if hasattr(scene, "compositing_node_group"):        # Blender >= 5.0
        tree = bpy.data.node_groups.new("simchange_comp", "CompositorNodeTree")
        scene.compositing_node_group = tree
        rl = tree.nodes.new("CompositorNodeRLayers")
        comp = tree.nodes.new("NodeGroupOutput")
        tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        tree.links.new(rl.outputs["Image"], comp.inputs[0])
    else:
        scene.use_nodes = True
        tree = scene.node_tree
        for n in list(tree.nodes):
            tree.nodes.remove(n)
        rl = tree.nodes.new("CompositorNodeRLayers")
        comp = tree.nodes.new("CompositorNodeComposite")
        tree.links.new(rl.outputs["Image"], comp.inputs["Image"])
    fo = tree.nodes.new("CompositorNodeOutputFile")
    if hasattr(fo, "directory"):                          # Blender >= 5.0 API
        fo.directory = str(depth_dir)
        fo.file_name = "depth_"
        fo.file_output_items.new("FLOAT", "depth")
        tree.links.new(rl.outputs["Depth"], fo.inputs["depth"])
        fo.format.file_format = "OPEN_EXR_MULTILAYER"
    else:
        fo.base_path = str(depth_dir)
        fo.file_slots[0].path = "depth_"
        tree.links.new(rl.outputs["Depth"], fo.inputs[0])
        fo.format.file_format = "OPEN_EXR"
        fo.format.color_mode = "RGB"
    fo.format.color_depth = "32"
    if cfg.get("_id_pass"):
        import hssd_scene
        hssd_scene.add_id_pass(scene, tree, rl, Path(str(depth_dir)))
    return fo


# ----------------------------------------------------------------------------- #
def top_level_objects(scene):
    return [o for o in scene.objects if o.parent is None]


def object_footprint_xy(o):
    return np.array(o.matrix_world.translation[:2])


def apply_variant(scene, cfg, spec, rng, meta):
    kind = spec["kind"]
    if kind in ("light",):
        f = spec["factor"]
        for o in scene.objects:
            if o.type == "LIGHT":
                o.data.energy *= f
        _scale_emission(f)
        _scale_world(scene, f)
        meta["light_factor"] = f
    elif kind == "lighting":
        apply_lighting(scene, cfg, spec["preset"], meta)
    elif kind == "combo":
        apply_lighting(scene, cfg, "evening", meta)
    elif kind == "multi" and spec.get("preset"):
        apply_lighting(scene, cfg, spec["preset"], meta)
    if kind == "move" or kind == "combo" or (kind == "multi" and spec.get("frac")):
        frac = spec.get("frac", 0.5)
        remove = spec.get("remove", False)
        cands = [o for o in top_level_objects(scene)
                 if any(p in o.name for p in cfg["movable"]) and not any(p in o.name for p in cfg["movable_exclude"])]
        cands.sort(key=lambda o: o.name)
        n = int(round(frac * len(cands)))
        chosen = list(rng.choice(len(cands), size=n, replace=False)) if n > 0 else []
        xmin, xmax, ymin, ymax = cfg["bounds"]
        changed = []
        for i in chosen:
            o = cands[i]
            if remove:
                o.hide_render = True
                o.hide_viewport = True
                for c in o.children_recursive:
                    c.hide_render = True
                changed.append({"name": o.name, "removed": True})
            else:
                dx, dy = rng.uniform(-1.5, 1.5, 2)
                p = o.matrix_world.translation
                nx, ny = float(np.clip(p.x + dx, xmin, xmax)), float(np.clip(p.y + dy, ymin, ymax))
                yaw = float(rng.uniform(-math.pi, math.pi))
                o.matrix_world = Matrix.Translation((nx, ny, p.z)) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Translation((-p.x, -p.y, -p.z)) @ o.matrix_world
                changed.append({"name": o.name, "dx": nx - p.x, "dy": ny - p.y, "yaw": yaw})
        meta["objects"] = {"candidates": len(cands), "changed": changed, "fraction": frac, "remove": remove}
    if kind == "rearrange" or (kind == "multi" and spec.get("level") is not None):
        import rearrange_plan
        import hssd_scene
        layout = json.loads(Path(cfg["layout"]).read_text())
        occ_path = meta.get("_occ_path")
        if occ_path is None:
            raise RuntimeError("rearrangement variants need --occ")
        base_xy, _, _ = build_path(cfg, {"kind": "map"}, np.load(occ_path))
        occ_low = Path(occ_path).with_name(Path(occ_path).name.replace("occ_", "occ_low_", 1))
        plan = rearrange_plan.make_plan(layout, occ_path, base_xy, spec["level"], spec.get("rseed", 0), ops=spec.get("rops"),
                                        occ_low_npz=str(occ_low) if occ_low.is_file() else None)
        applied, missing = hssd_scene.apply_plan(scene, plan)
        meta["rearrangement"] = {k: v for k, v in plan.items() if k not in ("changes", "pool_indices")}
        meta["rearrangement"]["applied"] = applied
        meta["rearrangement"]["missing"] = missing
        meta["_plan"] = plan
        print(f"[rearrange] level {spec['level']} seed {spec.get('rseed', 0)}: {applied} changes applied, {len(missing)} missing", flush=True)
    if kind == "background" or (kind == "multi" and spec.get("background")):
        changed = []
        palette = [(0.55, 0.62, 0.72, 1.0), (0.72, 0.55, 0.45, 1.0), (0.45, 0.6, 0.45, 1.0), (0.62, 0.58, 0.5, 1.0)]
        k = 0
        for o in scene.objects:
            if o.type != "MESH":
                continue
            is_wall = any(p == o.name.split(".")[0] for p in cfg["walls"]) or (cfg.get("walls_prefix") and o.name.startswith(cfg["walls_prefix"]))
            if not is_wall:
                continue
            for slot in o.material_slots:
                m = slot.material
                if m is None or not m.use_nodes:
                    continue
                for node in m.node_tree.nodes:
                    if node.type == "BSDF_PRINCIPLED":
                        inp = node.inputs["Base Color"]
                        for l in list(inp.links):
                            m.node_tree.links.remove(l)
                        inp.default_value = palette[k % len(palette)]
                        k += 1
                        changed.append(f"{o.name}/{m.name}")
        meta["background_changed"] = changed
    return meta


def _scale_emission(f):
    for m in bpy.data.materials:
        if not m.use_nodes:
            continue
        for node in m.node_tree.nodes:
            if node.type == "EMISSION" and not node.inputs["Strength"].is_linked:
                node.inputs["Strength"].default_value *= f
            elif node.type == "BSDF_PRINCIPLED" and "Emission Strength" in node.inputs and not node.inputs["Emission Strength"].is_linked:
                node.inputs["Emission Strength"].default_value *= f


def _scale_world(scene, f):
    w = scene.world
    if w is None or not w.use_nodes:
        return
    for node in w.node_tree.nodes:
        if node.type == "BACKGROUND" and not node.inputs["Strength"].is_linked:
            node.inputs["Strength"].default_value *= f


# ----------------------------------------------------------------------------- #
def read_exr_depth(path: Path) -> np.ndarray:
    """Read the depth channel of a (possibly multilayer) EXR written by the file output node."""
    import OpenEXR
    import Imath
    f = OpenEXR.InputFile(str(path))
    hdr = f.header()
    dw = hdr["dataWindow"]
    W, H = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    chans = list(hdr["channels"].keys())
    pick = [c for c in chans if c.endswith(".R") or c.endswith(".V") or c in ("R", "V", "Z")]
    ch = pick[0] if pick else chans[0]
    buf = f.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
    return np.frombuffer(buf, dtype=np.float32).reshape(H, W).copy()


def render_traversal(args, scene_name, variant, occ):
    cfg = SCENES[scene_name]
    spec = variant_spec(variant)
    out = (Path(args.out) / (variant + (getattr(args, "out_suffix", "") or ""))).resolve()
    (out / "left").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    tmp_depth = out / "_exr"
    tmp_depth.mkdir(exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(cfg["blend"]))
    scene = bpy.context.scene
    if cfg.get("unhide_all"):
        # demo files animate object visibility for their showcase video; render the complete scene
        for o in scene.objects:
            if "Hidden" in o.name:
                continue
            if o.animation_data is not None:
                o.animation_data_clear()
            o.hide_render = False
            o.hide_viewport = False
    for o in scene.objects:                                   # HSSD: door leaves are treated as open (hidden)
        if cfg.get("hide_categories") and o.get("hssd_category") in cfg["hide_categories"]:
            o.hide_render = True
            o.hide_viewport = True
    if cfg.get("fix_glass"):
        import hssd_scene
        n_glass = len(hssd_scene.fix_glass(scene))
        print(f"[{variant}] {n_glass} glass materials made shadow-transparent")
    if cfg.get("exposure_base") is not None:                   # indoor scenes: camera auto-exposure baseline
        scene.view_settings.exposure = float(cfg["exposure_base"])
    for name in cfg.get("exclude_collections", []):
        def _walk(lc):
            for ch in lc.children:
                if name in ch.name:
                    ch.exclude = True
                _walk(ch)
        _walk(scene.view_layers[0].layer_collection)
    rng = np.random.default_rng(args.seed)
    meta = {"scene": scene_name, "variant": variant, "spec": spec, "seed": args.seed, "_occ_path": args.occ}
    if getattr(args, "waypoints", None):
        cfg = dict(cfg, waypoints=[tuple(map(float, w.split(":"))) for w in args.waypoints.split(",")])
    path_spec = dict(spec)
    if spec["kind"] == "combo":
        path_spec.update({"reverse": True})
        spec.update({"frac": 0.5, "remove": False})
    apply_variant(scene, cfg, spec, rng, meta)
    if getattr(args, "exposure_override", None) is not None:
        scene.view_settings.exposure = float(args.exposure_override)
    meta["exposure_final"] = float(scene.view_settings.exposure)
    plan = meta.pop("_plan", None)
    meta.pop("_occ_path", None)
    xy, z, heading = build_path(cfg, path_spec, occ)
    if args.max_frames:
        xy, heading = xy[: args.max_frames], heading[: args.max_frames]
    if getattr(args, "dump_path", False):
        np.save(out / "path_xy.npy", xy)
        (out / "path_heading.npy").write_bytes(b"") if False else np.save(out / "path_heading.npy", heading)
        if plan is not None:
            (out / "plan.json").write_text(json.dumps(plan, indent=1))
        print(f"[{variant}] path dumped: {len(xy)} frames, {0.1 * len(xy):.1f} m")
    stride = args.preview_stride if args.preview else 1
    if getattr(args, "frame_stride", 1) > 1:
        stride = args.frame_stride
    idx = list(range(0, len(xy), stride))
    if getattr(args, "frames", None):
        idx = [int(t) for t in args.frames.split(",")]
    expo_per_station = None
    if cfg.get("auto_exposure") and cfg.get("layout"):
        # camera auto-exposure: darker exposure while the rig is in an outdoor region (smoothed over ~2 m)
        ae = cfg["auto_exposure"]
        lay = json.loads(Path(cfg["layout"]).read_text())
        polys = [np.array(r["poly"])[:, :2] for r in lay.get("regions", []) if r.get("poly") and len(r["poly"]) >= 3
                 and any(r["name"].startswith(n) for n in ae["regions"])]

        def _inside(pt, poly):
            x, y = pt; inside = False; j = len(poly) - 1
            for i in range(len(poly)):
                xi, yi = poly[i]; xj, yj = poly[j]
                if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
                    inside = not inside
                j = i
            return inside
        outdoor = np.array([1.0 if any(_inside(p, poly) for poly in polys) else 0.0 for p in xy])
        k = 21
        sm = np.convolve(np.pad(outdoor, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
        preset = meta.get("lighting", {}).get("preset", "default")
        off = ae["offset"].get(preset, ae["offset"]["default"])
        base = float(scene.view_settings.exposure)
        expo_per_station = base + off * sm
        meta["auto_exposure"] = {"base": base, "offset": off, "outdoor_fraction": float(outdoor.mean())}
    orig_index = np.arange(len(xy))
    if cfg.get("max_turn_deg_per_frame"):
        n_before = len(xy)
        xy, heading, orig_index = insert_turns(xy, heading, cfg["max_turn_deg_per_frame"])
        if expo_per_station is not None:                     # inserted stations take the exposure of the station they rotate at
            src = np.maximum.accumulate(np.where(orig_index >= 0, orig_index, 0))
            expo_per_station = np.asarray(expo_per_station)[src]
        meta["turn_frames"] = {"inserted": int((orig_index < 0).sum()), "max_turn_deg_per_frame": cfg["max_turn_deg_per_frame"], "n_before": n_before}
        print(f"[{variant}] {int((orig_index < 0).sum())} in-place rotation stations inserted ({n_before} -> {len(xy)})", flush=True)
        idx = list(range(0, len(xy), stride))
        if getattr(args, "frames", None):
            idx = [int(t) for t in args.frames.split(",")]
        if getattr(args, "turn_frames_only", False):
            idx = [int(k) for k in np.nonzero(orig_index < 0)[0]]
    if getattr(args, "dump_path", False):
        (out / "stations.json").write_text(json.dumps({"orig_index": orig_index.tolist(), "n_stations": len(xy)}))
        poses_all = [c2w_opencv(camera_matrix_blender(xy[i, 0], xy[i, 1], z, heading[i])).reshape(-1) for i in range(len(xy))]
        np.savetxt(out / "poses_left_all.txt", np.asarray(poses_all), fmt="%.8f")
        np.save(out / "path_xy.npy", xy)
        np.save(out / "path_heading.npy", heading)
        if args.dump_path_only:
            print(f"[{variant}] stations after turn insertion: {len(xy)} ({int((orig_index < 0).sum())} rotation frames)")
            return
    if occ is not None:
        cl = clearance(xy, occ)
        meta["clearance_min"] = float(cl.min())
        meta["clearance_p05"] = float(np.percentile(cl, 5))
        print(f"[{variant}] {len(xy)} frames, clearance min {cl.min():.2f} m, 5% {np.percentile(cl, 5):.2f} m")

    baselines = [float(b) for b in args.baselines.split(",") if b.strip()]
    cams, K = setup_rig(scene, args.width, args.height, args.fov, baselines)
    if getattr(args, "engine", None):
        cfg = dict(cfg, engine=args.engine)
    if getattr(args, "id_pass", False):
        cfg = dict(cfg, _id_pass=True)
        (out / "ids").mkdir(exist_ok=True)
    fo = setup_render(scene, cfg, args.samples or cfg["samples"], tmp_depth)
    if getattr(args, "quantify", False):            # id/depth passes only: 1 sample, no denoising
        if scene.render.engine == "CYCLES":
            scene.cycles.samples = 1
            scene.cycles.use_denoising = False
            scene.cycles.use_adaptive_sampling = False
            scene.cycles.max_bounces = 0
    frames_dir = out / "_frames"
    frames_dir.mkdir(exist_ok=True)
    for b in baselines:
        (out / f"right_{b:.2f}").mkdir(exist_ok=True)
    # keyframe all cameras: animation frame k*(1+nB) = left view of station k, then one right view per baseline
    nb = len(baselines)
    stride_f = 1 + nb
    poses = []
    scene.timeline_markers.clear()
    for cam in cams.values():
        cam.animation_data_clear()
        cam.rotation_mode = "XYZ"
    for k, i in enumerate(idx):
        M = camera_matrix_blender(xy[i, 0], xy[i, 1], z, heading[i])
        mats = [("simL", M)] + [(f"simR_{b:.2f}", M @ Matrix.Translation((b, 0.0, 0.0))) for b in baselines]
        for j in range(stride_f):
            f = k * stride_f + j
            for name, Mc in mats:
                cams[name].matrix_world = Mc
                cams[name].keyframe_insert("location", frame=f)
                cams[name].keyframe_insert("rotation_euler", frame=f)
            scene.timeline_markers.new(f"{mats[j][0]}_{k}", frame=f).camera = cams[mats[j][0]]
            if expo_per_station is not None:
                scene.view_settings.exposure = float(expo_per_station[i])
                scene.keyframe_insert(data_path="view_settings.exposure", frame=f)
        poses.append(c2w_opencv(M).reshape(-1))
    scene.frame_start, scene.frame_end = 0, stride_f * len(idx) - 1
    scene.camera = cams["simL"]
    scene.render.filepath = str(frames_dir / "####")
    t0 = time.time()
    bpy.ops.render.render(animation=True, write_still=False)
    # distribute frames (one directory listing instead of a glob per station: the output may live on a network share)
    t1 = time.time()
    exr_by_frame, id_by_frame = {}, {}
    for e in tmp_depth.iterdir():
        if e.suffix != ".exr":
            continue
        try:
            fnum = int(e.stem[-4:])
        except ValueError:
            continue
        (id_by_frame if e.name.startswith("id_") else exr_by_frame)[fnum] = e
    for k in range(len(idx)):
        f0 = k * stride_f
        (frames_dir / f"{f0:04d}.png").rename(out / "left" / f"{k:06d}.png")
        for j, b in enumerate(baselines):
            (frames_dir / f"{f0 + 1 + j:04d}.png").rename(out / f"right_{b:.2f}" / f"{k:06d}.png")
        if f0 not in exr_by_frame:
            raise RuntimeError(f"no depth EXR for frame {f0} in {tmp_depth}")
        d = read_exr_depth(exr_by_frame[f0])
        d[d > 90.0] = 0.0
        np.save(out / "depth" / f"{k:06d}.npy", d.astype(np.float16))
        if getattr(args, "id_pass", False) and f0 in id_by_frame:
            ids = read_exr_depth(id_by_frame[f0])
            np.save(out / "ids" / f"{k:06d}.npy", np.rint(ids).astype(np.int32))
    print(f"[{variant}] frames distributed in {time.time() - t1:.0f}s", flush=True)
    for f in tmp_depth.glob("*.exr"):
        f.unlink()
    frames_dir.rmdir()
    # default "right" = smallest baseline
    right_link = out / "right"
    if baselines:
        if right_link.is_symlink() or right_link.exists():
            right_link.unlink() if right_link.is_symlink() else None
        if not right_link.exists():
            right_link.symlink_to(f"right_{baselines[0]:.2f}")
    np.savetxt(out / "poses_left.txt", np.asarray(poses), fmt="%.8f")
    if plan is not None:
        (out / "plan.json").write_text(json.dumps(plan, indent=1))
    if idx and (len(idx) != len(xy)):
        np.savetxt(out / "frame_index.txt", np.asarray(idx), fmt="%d")
    T_rl = np.eye(4)
    T_rl[0, 3] = baselines[0] if baselines else 0.0
    calib = {"K": K.tolist(), "width": args.width, "height": args.height, "baseline": baselines[0] if baselines else 0.0,
             "baselines": baselines, "right_dirs": {f"{b:.2f}": f"right_{b:.2f}" for b in baselines},
             "T_right_in_left": T_rl.tolist(), "fps": args.fps, "scene": scene_name, "variant": variant,
             "step_m": 0.1, "n_frames": len(idx), "world_up": "z"}
    if cfg.get("max_turn_deg_per_frame"):
        calib["turn_frames_inserted"] = int((orig_index < 0).sum())
        calib["max_turn_deg_per_frame"] = cfg["max_turn_deg_per_frame"]
    (out / "calib.json").write_text(json.dumps(calib, indent=1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    try:
        tmp_depth.rmdir()
    except OSError:
        pass
    print(f"[{variant}] done: {len(idx)} frames in {time.time() - t0:.0f}s")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=list(SCENES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="map")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--baselines", default="0.1,0.3,0.5", help="stereo baselines (m); one right image per baseline")
    ap.add_argument("--gpu-index", type=int, default=-1, help="CUDA device index for Cycles (-1: all)")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--engine", default=None, help="override the scene's render engine (CYCLES | BLENDER_EEVEE)")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--preview-stride", type=int, default=10)
    ap.add_argument("--occ", default=None, help="occupancy npz for clearance report")
    ap.add_argument("--waypoints", default=None, help="override scene waypoints: x:y,x:y,...")
    ap.add_argument("--dump-path", action="store_true", help="write path_xy.npy / plan.json into the variant dir")
    ap.add_argument("--dump-path-only", action="store_true", help="with --dump-path: do not render")
    ap.add_argument("--id-pass", action="store_true", help="also write the object-index pass (ids/*.npy)")
    ap.add_argument("--quantify", action="store_true", help="passes only (1 sample, no denoise); combine with --id-pass")
    ap.add_argument("--frame-stride", type=int, default=1, help="render every n-th station")
    ap.add_argument("--frames", default=None, help="explicit comma-separated station indices to render")
    ap.add_argument("--exposure-override", type=float, default=None, help="force the view exposure (lighting tests)")
    ap.add_argument("--turn-frames-only", action="store_true", help="render only the inserted in-place rotation stations (splice later)")
    ap.add_argument("--out-suffix", default="", help="append to the variant output directory name")
    args = ap.parse_args(argv)
    if args.dump_path_only:
        args.dump_path = True
    global ARGS_GPU_INDEX
    ARGS_GPU_INDEX = args.gpu_index
    occ = np.load(args.occ) if args.occ else None
    for v in args.variants.split(","):
        render_traversal(args, args.scene, v, occ)


if __name__ == "__main__":
    main()
