"""Render the trajectories of a scene with Cycles into the scene cache (run with the bpy environment).

    python -m cross_nav.sim.render_seq --blend <scene.blend> --traj <traj.npz> --cache <root> --dataset hssd_nav \
        --scene <name> [--samples 32] [--tmp /data0/jz/navsim/tmp]

Reuses SimChange's renderer (simchange/render.py, imported from $SIMCHANGE_ROOT): its lighting presets (sun / sky /
interior lamps), glass fix and the Cycles + depth-pass setup.  One trajectory = one sequence (its own session) of one
SceneWriter scene; every trajectory re-opens the blend file (lighting changes the scene).  A per-trajectory exposure is
chosen from a quick low-resolution pre-render of a few frames (camera auto-exposure).  Depth = the Cycles Z pass
(distance along the optical axis); E = camera-from-world in OpenCV axes, world = the blend's z-up frame.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def simchange_render():
    root = os.environ.get("SIMCHANGE_ROOT", "/data2/projects/cross_v2/SimChange")
    sys.path.insert(0, os.path.join(root, "simchange"))
    import render as R                                       # noqa: E402  (SimChange; imports bpy)
    return R


HSSD_CFG = {
    "engine": "CYCLES", "samples": 32, "hide_categories": ["door"], "fix_glass": True,
    "sky_nishita": True, "sun_light": None, "sun_energy": 1.0, "window_azimuth_deg": 0.0,
    "lamp_prop": "hssd_lamp", "torch_objects": None, "torch_energy": 40.0,
    "portal_materials": [], "portal_strength": 0.0, "lamp_lights": [], "lamp_materials": [], "fill_lights": [],
    "exposure_base": 3.0,
    "exposure": {"morning": 0.3, "noon": 0.0, "afternoon": 0.2, "evening": -2.0, "night": 0.0, "overcast": 0.5},
}


def prepare_scene(R, blend: str, cfg: dict, preset: str, azimuth: float):
    import bpy
    bpy.ops.wm.open_mainfile(filepath=blend)
    scene = bpy.context.scene
    for o in scene.objects:
        if o.get("hssd_category") in cfg["hide_categories"]:
            o.hide_render = True
            o.hide_viewport = True
    if cfg.get("fix_glass"):
        import hssd_scene
        hssd_scene.fix_glass(scene)
    scene.view_settings.exposure = float(cfg["exposure_base"])
    meta = {}
    R.apply_lighting(scene, dict(cfg, window_azimuth_deg=azimuth), preset, meta)
    return scene, meta


def make_camera(scene, width, height, K):
    import bpy
    import math
    data = bpy.data.cameras.new("navcam")
    data.sensor_fit = "HORIZONTAL"
    data.lens_unit = "FOV"
    data.angle = 2 * math.atan(width / 2.0 / K[0, 0])
    data.clip_start = 0.02
    data.clip_end = 100.0
    cam = bpy.data.objects.new("navcam", data)
    scene.collection.objects.link(cam)
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    scene.camera = cam
    return cam


def keyframe(scene, cam, c2w):
    from mathutils import Matrix
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    cam.animation_data_clear()
    cam.rotation_mode = "XYZ"
    for k, T in enumerate(c2w):
        cam.matrix_world = Matrix((T @ flip).tolist())
        cam.keyframe_insert("location", frame=k)
        cam.keyframe_insert("rotation_euler", frame=k)
    # constant interpolation: frames are independent stations
    ad = cam.animation_data
    if ad and ad.action:
        try:
            for fc in ad.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"
        except AttributeError:                         # Blender 5 layered actions
            pass
    scene.frame_start, scene.frame_end = 0, len(c2w) - 1


def auto_exposure(R, scene, cam, c2w, cfg, tmp: Path, target=0.42):
    """Exposure offset (stops) that brings the median luminance of a few low-resolution, low-sample renders to target."""
    import bpy
    import cv2
    sel = np.linspace(0, len(c2w) - 1, min(6, len(c2w))).round().astype(int)
    rx, ry, pct = scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage
    scene.render.resolution_percentage = 25
    samples = scene.cycles.samples
    scene.cycles.samples = 8
    lum = []
    for i in sel:
        keyframe(scene, cam, c2w[i:i + 1])
        scene.render.filepath = str(tmp / f"ae_{i}.png")
        bpy.ops.render.render(write_still=True)
        img = cv2.imread(str(tmp / f"ae_{i}.png"), cv2.IMREAD_COLOR)
        if img is not None:
            lum.append(np.median(img.astype(np.float32).mean(2)) / 255.0)
    scene.render.resolution_percentage = pct
    scene.cycles.samples = samples
    if not lum:
        return 0.0
    m = float(np.median(lum))
    # display transform is roughly gamma 1/2.2 on linear exposure: one stop = x 2^(1/2.2) in display value
    off = float(np.clip(2.2 * np.log2(target / max(m, 1e-3)), -4.0, 4.0))
    return off


def render_traj(R, args, tr: dict, sw, tmp_root: Path):
    import bpy
    import cv2
    t0 = time.time()
    cfg = dict(HSSD_CFG)
    scene, meta = prepare_scene(R, args.blend, cfg, tr["preset"], tr.get("azimuth", 0.0))
    c2w, K, W, H = tr["c2w"], tr["K"], int(tr["width"]), int(tr["height"])
    tmp = Path(tempfile.mkdtemp(prefix=f"{args.scene}_{tr['name']}_", dir=tmp_root))
    depth_dir = tmp / "_exr"
    depth_dir.mkdir()
    cam = make_camera(scene, W, H, K)
    R.setup_render(scene, cfg, args.samples, depth_dir)
    scene.render.image_settings.compression = 15
    expo = auto_exposure(R, scene, cam, c2w, cfg, tmp) if args.auto_exposure else 0.0
    scene.view_settings.exposure = float(scene.view_settings.exposure) + expo
    for f in depth_dir.iterdir():
        f.unlink()
    keyframe(scene, cam, c2w)
    frames = tmp / "_frames"
    frames.mkdir()
    scene.render.filepath = str(frames / "####")
    t1 = time.time()
    bpy.ops.render.render(animation=True, write_still=False)
    t2 = time.time()
    exr = {}
    for e in depth_dir.iterdir():
        if e.suffix == ".exr":
            try:
                exr[int(e.stem[-4:])] = e
            except ValueError:
                pass
    q = sw.sequence(tr["name"], session=tr["name"])
    for k in range(len(c2w)):
        img = cv2.imread(str(frames / f"{k:04d}.png"), cv2.IMREAD_COLOR)
        if img is None or k not in exr:
            print(f"[render] {tr['name']} frame {k} missing", flush=True)
            continue
        d = R.read_exr_depth(exr[k])
        d[~np.isfinite(d) | (d > 60.0) | (d <= 0)] = 0.0
        E = np.linalg.inv(c2w[k])
        q.add(f"{k:06d}", cv2.cvtColor(img, cv2.COLOR_BGR2RGB), d, K, E)
    shutil.rmtree(tmp, ignore_errors=True)
    info = {"name": tr["name"], "frames": len(c2w), "preset": tr["preset"], "exposure": float(scene.view_settings.exposure),
            "auto_exposure_offset": expo, "render_s": round(t2 - t1, 1), "total_s": round(time.time() - t0, 1),
            "s_per_frame": round((t2 - t1) / max(len(c2w), 1), 2), "lamps": meta.get("n_lamp_lights", 0)}
    print("[render]", json.dumps(info), flush=True)
    return info


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--traj", required=True, help="npz written by pipeline.py (trajectories of the scene)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset", default="hssd_nav")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--world", default=None)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--tmp", default=os.environ.get("NAVSIM_TMP", "/tmp"))
    ap.add_argument("--no-auto-exposure", dest="auto_exposure", action="store_false")
    ap.add_argument("--only", nargs="*", default=None, help="trajectory names")
    ap.add_argument("--info", default=None, help="write render statistics (json) here")
    a = ap.parse_args(argv)
    R = simchange_render()
    from vggt_ft.prep.common import SceneWriter
    z = np.load(a.traj, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    trajs = []
    for t in meta["trajs"]:
        if a.only and t["name"] not in a.only:
            continue
        trajs.append(dict(t, c2w=z[f"c2w/{t['name']}"], K=np.array(t["K"])))
    tmp_root = Path(a.tmp)
    tmp_root.mkdir(parents=True, exist_ok=True)
    sw = SceneWriter(a.cache, a.dataset, a.scene, world=a.world or a.scene, metric=True, synthetic=True, kind="indoor",
                     extra={"source": meta.get("source", "hssd"), "render": "blender cycles (SimChange)"})
    infos = [render_traj(R, a, tr, sw, tmp_root) for tr in trajs]
    n = sw.close()
    print(f"[render] scene {a.scene}: {n} frames in {len(infos)} sequences", flush=True)
    if a.info:
        Path(a.info).write_text(json.dumps(infos, indent=1))


if __name__ == "__main__":
    main()
