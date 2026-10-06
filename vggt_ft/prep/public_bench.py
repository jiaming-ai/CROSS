"""Public benchmarks as held-out test scenes (never trained on).  One function per dataset; each reads the official
release (or the DA3-BENCH repack where noted) from <raw>/<dataset> and writes <out>/<dataset>/<scene>/... with
scene.json "split": "test", "source" and "depth_src".  Every sequence keeps at most --max-frames frames (constant
stride).  Extrinsics are camera-from-world, OpenCV axes; depth is z-depth (checked: multi-view consistency, planes).

  eth3d     13 high-res multi-view training scenes: undistorted DSLR JPGs + COLMAP poses (metres); the laser depth
            and occlusion masks belong to the distorted images and are resampled into the undistorted ones (sparse)
  7scenes   7 scenes, all sequences (train + test) of a scene as sessions in its one KinectFusion frame; raw depth
            registered to the RGB camera, RGB-camera poses
  nrgbd     Neural RGB-D synthetic scenes (Azinovic et al. 2022): rendered RGB + noise-free depth, OpenGL poses
  dtu       22 MVSNet evaluation scans (DA3-BENCH dtu.zip: Rectified, Cameras, Depths_raw); DTU's mm -> metres
  hiroom    DA3's synthetic HiRoom (DA3-BENCH hiroom.zip): rendered depth (uint16 / 65535 * 100 m), aliasing masks
  nyuv2     654 Eigen test images (labelled set, in-painted depth), Eigen crop; single views grouped per room type
  diode     DIODE val (indoor + outdoor laser depth); single views grouped per scene
  sintel    MPI Sintel training (final pass, depth + camera), synthetic, dynamic, not metric
  tum_dynamic  TUM RGB-D fr3 walking_* (dynamic); registered depth / 5000, mocap poses, one scene (mocap frame)
  bonn_dynamic Bonn RGB-D Dynamic, 5 sequences (dynamic); same format as tum, marker-to-camera calibration applied

Single-view sets (nyuv2, diode) store identity poses and kind "single_view": no pose / covisibility supervision.

Raw layout (archives extracted under <raw_root>): eth3d/x/<scene>/ (undistorted images + calibration, depth, occlusion
masks, and dslr_calibration_jpg/cameras.txt from <scene>_dslr_jpg.7z), 7scenes/x/<scene>/seq-XX/, nrgbd/x/<scene>/,
da3/dtu/, da3/hiroom/, nyu/{nyu_depth_v2_labeled.mat,splits.mat}, diode/val/, sintel/training/,
tum/rgbd_dataset_freiburg3_walking_*/, bonn/rgbd_bonn_*/.

  python -m vggt_ft.prep.public_bench <dataset> <raw_root> <out_root> [--max-frames 300] [--workers 16]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from vggt_ft.prep.common import SceneWriter, units_for, write_index

MAX_FRAMES = 300


def stride(n: int, m: int | None = None) -> np.ndarray:
    m = m or MAX_FRAMES
    return np.arange(0, n, int(np.ceil(n / m)))


def rgb_of(p) -> np.ndarray:
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def inv(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, np.float64)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def qvec2R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def writer(out, dataset, scene, source, depth_src, **kw) -> SceneWriter:
    extra = dict(split="test", source=source, depth_src=depth_src, **kw.pop("extra", {}))
    return SceneWriter(out, dataset, scene, extra=extra, **kw)


# ----------------------------------------------------------------------------------------------------- NRGBD
def nrgbd(raw: Path, out: Path, workers: int):
    base = raw / "nrgbd" / "x"
    jobs = [(d, out) for d in sorted(base.iterdir()) if (d / "poses.txt").is_file()]
    return _run(_nrgbd_scene, jobs, workers)


def _nrgbd_scene(d: Path, out: Path):
    """images/img<i>.png, depth/depth<i>.png (synthetic ground truth, mm), poses.txt (camera-to-world, OpenGL axes,
    4 rows per frame, nan = invalid), focal.txt; principal point at the image centre (NeRF convention)."""
    rows = np.array([[float(x) for x in ln.split()] for ln in (d / "poses.txt").read_text().splitlines()
                     if ln.strip()])
    poses = rows.reshape(-1, 4, 4)
    f = float((d / "focal.txt").read_text().split()[0])
    sw = writer(out, "nrgbd", d.name, "https://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip",
                "rendered (depth/, noise-free)", metric=True, synthetic=True, kind="indoor")
    q = sw.sequence("traj")
    ok = [i for i in range(len(poses)) if np.isfinite(poses[i]).all() and (d / "images" / f"img{i}.png").exists()]
    for i in [ok[k] for k in stride(len(ok))]:
        rgb = rgb_of(d / "images" / f"img{i}.png")
        H, W = rgb.shape[:2]
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
        dep = cv2.imread(str(d / "depth" / f"depth{i}.png"), cv2.IMREAD_UNCHANGED) / 1000.0
        q.add(f"{i:05d}", rgb, dep, K, inv(poses[i] @ np.diag([1.0, -1, -1, 1])))
    return sw.close()


# ----------------------------------------------------------------------------------------------------- HiRoom
def hiroom(raw: Path, out: Path, workers: int):
    base = raw / "da3" / "hiroom" / "data"
    jobs = [(d, out) for d in sorted(base.glob("*/*/cam_sampled_*"))]
    return _run(_hiroom_scene, jobs, workers)


def _hiroom_scene(d: Path, out: Path):
    sw = writer(out, "hiroom", f"{d.parent.name}_{d.name[-2:]}", "https://huggingface.co/datasets/depth-anything/"
                "DA3-BENCH (hiroom.zip)", "rendered; aliasing_mask -> invalid", metric=True, synthetic=True,
                kind="indoor")
    K = np.load(d / "cam_K.npy")
    q = sw.sequence("views")
    names = sorted((p.stem for p in (d / "image").glob("*.jpg")), key=int)
    for i in stride(len(names)):
        n = names[i]
        dep = cv2.imread(str(d / "depth" / f"{n}.png"), cv2.IMREAD_UNCHANGED) / 65535.0 * 100.0
        m = cv2.imread(str(d / "aliasing_mask" / f"{n}.png"), cv2.IMREAD_UNCHANGED)
        if m is not None:
            dep[m > 0] = 0
        q.add(n.zfill(4), rgb_of(d / "image" / f"{n}.jpg"), dep, K, np.load(d / "pose" / f"{n}.npy"))
    return sw.close()


# ----------------------------------------------------------------------------------------------------- DTU
def read_pfm(p) -> np.ndarray:
    with open(p, "rb") as f:
        f.readline()
        w, h = map(int, f.readline().split())
        s = float(f.readline())
        return np.flipud(np.fromfile(f, "<f4" if s < 0 else ">f4").reshape(h, w)).copy()


def dtu(raw: Path, out: Path, workers: int):
    base = raw / "da3" / "dtu"
    return _run(_dtu_scene, [(d, out) for d in sorted((base / "Rectified").iterdir())], workers)


def _dtu_scene(d: Path, out: Path):
    """Images rect_<k>_3_r5000 (lighting 3) <-> Cameras/<k-1>_cam.txt (camera-from-world, mm) <-> Depths_raw
    depth_map_<k-1>.pfm (mm, rendered from the reference mesh; 0 = no surface)."""
    base = d.parent.parent
    sw = writer(out, "dtu", d.name, "https://huggingface.co/datasets/depth-anything/DA3-BENCH (dtu.zip; MVSNet "
                "Rectified + Cameras + Depths_raw)", "MVSNet Depths_raw (rendered from the DTU reference mesh)",
                metric=True, kind="object", extra=dict(note="DTU calibration is in mm; stored in metres"))
    q = sw.sequence("rect")
    files = sorted(d.glob("rect_*_3_r5000.png"))
    for i in stride(len(files)):
        k = int(files[i].name.split("_")[1]) - 1
        L = (base / "Cameras" / f"{k:08d}_cam.txt").read_text().split()
        E = np.array(L[1:17], np.float64).reshape(4, 4)
        K = np.array(L[18:27], np.float64).reshape(3, 3)
        E[:3, 3] /= 1000.0
        dep = read_pfm(base / "depth_raw" / "Depths" / d.name / f"depth_map_{k:04d}.pfm") / 1000.0
        q.add(f"{k:04d}", rgb_of(files[i]), dep, K, E)
    return sw.close()


# ----------------------------------------------------------------------------------------------------- ETH3D
def eth3d(raw: Path, out: Path, workers: int):
    base = raw / "eth3d" / "x"
    jobs = [(d, out) for d in sorted(base.iterdir()) if (d / "dslr_calibration_undistorted").is_dir()]
    return _run(_eth3d_scene, jobs, min(workers, 13))


def _colmap_cams(p: Path) -> dict:
    cams = {}
    for ln in p.read_text().splitlines():
        if ln and not ln.startswith("#"):
            c = ln.split()
            cams[c[0]] = (int(c[2]), int(c[3]), np.array(c[4:], np.float64))
    return cams


def _thin_prism(x, y, k):
    """COLMAP THIN_PRISM_FISHEYE distortion of normalised coordinates (k1 k2 p1 p2 k3 k4 sx1 sy1)."""
    k1, k2, p1, p2, k3, k4, sx1, sy1 = k
    r = np.hypot(x, y)
    th = np.arctan(r)
    t2 = th * th
    s = np.where(r > 1e-12, th * (1 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4) / np.maximum(r, 1e-12), 1)
    u, v = x * s, y * s
    rr = u * u + v * v
    return (u + 2 * p1 * u * v + p2 * (rr + 2 * u * u) + sx1 * rr,
            v + 2 * p2 * u * v + p1 * (rr + 2 * v * v) + sy1 * rr)


def _eth3d_depth(dep_d, cam_d, W, H, ku, max_side=768):
    """Ground-truth depth is given for the distorted images (6048x4032, sparse laser points, inf = none): resample it
    at the pixel centres of the undistorted image as stored (long side max_side), taking the nearest valid sample
    within the output pixel's footprint."""
    s = min(1.0, max_side / max(W, H))
    nw, nh = int(round(W * s)), int(round(H * s))
    fx, fy, cx, cy = ku                                  # COLMAP convention: pixel centres at +0.5
    xs = ((np.arange(nw) + 0.5) * W / nw - cx) / fx
    ys = ((np.arange(nh) + 0.5) * H / nh - cy) / fy
    xn, yn = np.meshgrid(xs, ys)
    xd, yd = _thin_prism(xn, yn, cam_d[2][4:12])
    fxd, fyd, cxd, cyd = cam_d[2][:4]
    X, Y = fxd * xd + cxd, fyd * yd + cyd                # distorted image, COLMAP coordinates
    Hd, Wd = dep_d.shape
    r = int(np.ceil(0.5 * W / nw))
    offs = sorted(((dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)), key=lambda o: o[0] ** 2 + o[1] ** 2)
    out = np.zeros((nh, nw), np.float32)
    for dx, dy in offs:
        ix, iy = np.floor(X).astype(int) + dx, np.floor(Y).astype(int) + dy
        ok = (out == 0) & (ix >= 0) & (ix < Wd) & (iy >= 0) & (iy < Hd)
        v = dep_d[iy[ok], ix[ok]]
        sub = out[ok]
        sub[np.isfinite(v) & (v > 0)] = v[np.isfinite(v) & (v > 0)]
        out[ok] = sub
    return out


def _eth3d_scene(d: Path, out: Path):
    """Undistorted images with their COLMAP model (PINHOLE; images.txt = camera-from-world, OpenCV axes, metres).
    The laser depth maps and the occlusion masks (masks_for_images: 1 / 2 = unreliable ground truth) belong to the
    distorted images (THIN_PRISM_FISHEYE, dslr_calibration_jpg): masked, then resampled into the undistorted view."""
    cams_u = _colmap_cams(d / "dslr_calibration_undistorted" / "cameras.txt")
    cams_d = _colmap_cams(d / "dslr_calibration_jpg" / "cameras.txt")
    rows = [ln for ln in (d / "dslr_calibration_undistorted" / "images.txt").read_text().splitlines()
            if not ln.startswith("#")][0::2]
    ims = []
    for r in rows:
        p = r.split()
        if len(p) >= 10:
            E = np.eye(4)
            E[:3, :3] = qvec2R(np.array(p[1:5], np.float64))
            E[:3, 3] = np.array(p[5:8], np.float64)
            ims.append((p[9], p[8], E))
    ims.sort()
    sw = writer(out, "eth3d", d.name, "https://www.eth3d.net/datasets (multi_view_training_dslr_undistorted.7z, "
                "multi_view_training_dslr_occlusion.7z, <scene>_dslr_depth.7z)",
                "laser scan (ground_truth_depth of the distorted images, occlusion masks applied, resampled into the "
                "undistorted images; sparse)", metric=True,
                kind="indoor" if d.name in ("delivery_area", "kicker", "office", "pipes", "relief", "relief_2",
                                            "terrains") else "outdoor")
    q = sw.sequence("dslr")
    for i in stride(len(ims)):
        name, cid, E = ims[i]
        W, H, ku = cams_u[cid]
        cd = cams_d[cid]
        stem = Path(name).stem
        dep = np.fromfile(d / "ground_truth_depth" / "dslr_images" / f"{stem}.JPG", np.float32).reshape(cd[1], cd[0])
        m = cv2.imread(str(d / "masks_for_images" / "dslr_images" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if m is not None:
            dep[m > 0] = np.inf
        K = np.array([[ku[0], 0, ku[2] - 0.5], [0, ku[1], ku[3] - 0.5], [0, 0, 1]])
        q.add(stem, rgb_of(d / "images" / name), _eth3d_depth(dep, cd, W, H, ku), K, E)
    return sw.close()


# ----------------------------------------------------------------------------------------------------- 7-Scenes
SEVEN = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
D_TO_RGB = np.array([[9.9996518012567637e-01, 2.6765126468950343e-03, -7.9041012313000904e-03, -2.5558943178152542e-02],
                     [-2.7409311281316700e-03, 9.9996302803027592e-01, -8.1504520778013286e-03, 1.0109636268061706e-04],
                     [7.8819942130445332e-03, 8.1718328771890631e-03, 9.9993554558014031e-01, 2.0318321729487039e-03],
                     [0, 0, 0, 1]])   # Kinect depth -> RGB sensor (ACE / DSAC*, generic LIRIS calibration)
D_TO_RGB[:3, 3] *= 0.65   # this sensor's offset: fitted to the DSAC* mesh renderings (they match the RGB edges)
SEVEN_K = np.array([[525.0, 0, 320], [0, 525.0, 240], [0, 0, 1]])


def register_depth(d_raw: np.ndarray, f_d: float = 585.0) -> np.ndarray:
    """Raw Kinect depth (m, depth camera, f 585) -> z-depth in the RGB camera (f 525), z-buffered splat."""
    H, W = d_raw.shape
    ys, xs = np.nonzero((d_raw > 0) & (d_raw < 100))
    z = d_raw[ys, xs]
    P = np.stack([(xs - W / 2) / f_d * z, (ys - H / 2) / f_d * z, z], 1) @ D_TO_RGB[:3, :3].T + D_TO_RGB[:3, 3]
    u = np.round(P[:, 0] / P[:, 2] * SEVEN_K[0, 0] + SEVEN_K[0, 2]).astype(int)
    v = np.round(P[:, 1] / P[:, 2] * SEVEN_K[1, 1] + SEVEN_K[1, 2]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (P[:, 2] > 0)
    out = np.full(H * W, np.inf, np.float32)
    np.minimum.at(out, v[ok] * W + u[ok], P[ok, 2].astype(np.float32))
    out[~np.isfinite(out)] = 0
    return out.reshape(H, W)


def sevenscenes(raw: Path, out: Path, workers: int):
    return _run(_seven_scene, [(s, raw / "7scenes", out) for s in SEVEN], workers)


def _seven_scene(scene: str, base: Path, out: Path):
    """All sequences of a scene (train and test splits) as sessions of one KinectFusion world frame.  The released
    poses are the depth sensor's and the depth is not registered to the RGB images: RGB pose = pose @ inv(D_TO_RGB),
    raw depth registered into the RGB camera with D_TO_RGB (the DSAC* / ACE mesh renderings cover only the training
    sequences; they served to fit D_TO_RGB's translation, the generic one leaves the depth edges ~7 px off)."""
    sw = writer(out, "7scenes", scene, "https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/",
                "raw Kinect depth registered to the RGB camera (f 585 -> 525, depth->RGB offset fitted to the DSAC* "
                "renderings, heiDATA doi:10.11588/data/N07HKC/4PLEEJ)", metric=True, kind="indoor",
                extra=dict(poses="RGB camera: original (depth sensor) pose @ inv(D_TO_RGB)",
                           d_to_rgb=D_TO_RGB.tolist()))
    for sd in [p for p in sorted((base / "x" / scene).glob("seq-*")) if p.is_dir()]:
        names = sorted(p.name[:-10] for p in sd.glob("frame-*.color.png"))
        q = sw.sequence(sd.name)
        for i in stride(len(names)):
            n = names[i]
            d = cv2.imread(str(sd / f"{n}.depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
            d[d >= 65535] = 0
            c2w = np.loadtxt(sd / f"{n}.pose.txt") @ inv(D_TO_RGB)
            q.add(n[6:], rgb_of(sd / f"{n}.color.png"), register_depth(d / 1000.0), SEVEN_K, inv(c2w))
    return sw.close()


# ----------------------------------------------------------------------------------------------------- single views
def _single_views(out, dataset, scene, items, K, source, depth_src, place):
    """items: [(name, rgb_fn, depth_fn)] of one place / room type; identity poses, kind single_view, chunks of
    at most MAX_FRAMES frames as sequences."""
    sw = writer(out, dataset, scene, source, depth_src, metric=True, kind="single_view",
                extra=dict(place=place, poses="identity (single views; no pose / covisibility supervision)"))
    for c in range(0, len(items), MAX_FRAMES):
        q = sw.sequence(f"views{c // MAX_FRAMES}" if len(items) > MAX_FRAMES else "views")
        for name, rgb_fn, dep_fn in items[c:c + MAX_FRAMES]:
            q.add(name, rgb_fn(), dep_fn(), K, np.eye(4))
    return sw.close()


NYU_K = np.array([[518.8579, 0, 325.5824 - 41], [0, 519.4696, 253.7362 - 45], [0, 0, 1]])   # after the Eigen crop


def nyuv2(raw: Path, out: Path, workers: int):
    """Labelled set (h5 v7.3), official test split = Eigen's 654 images; in-painted depth (the standard GT);
    image and depth cropped to Eigen's crop [45:471, 41:601] (removes the white border of the registration)."""
    import h5py
    from scipy.io import loadmat
    f = h5py.File(raw / "nyu" / "nyu_depth_v2_labeled.mat", "r")
    test = loadmat(raw / "nyu" / "splits.mat")["testNdxs"].ravel() - 1
    txt = lambda r: "".join(chr(c) for c in f[r][:].ravel())
    loc = [txt(r) for r in f["scenes"][0]]
    typ = [txt(r) for r in f["sceneTypes"][0]]
    groups = {}
    for i in sorted(test, key=lambda i: (loc[i], i)):
        groups.setdefault(typ[i], []).append(i)
    small = [t for t, v in groups.items() if len(v) < 2]
    for t in small:
        groups.setdefault("other", []).extend(groups.pop(t))
    n = 0
    for t, idx in groups.items():
        items = [(f"{loc[i]}_{i:04d}", lambda i=i: f["images"][i].transpose(2, 1, 0)[45:471, 41:601],
                  lambda i=i: f["depths"][i].T[45:471, 41:601]) for i in idx]
        n += _single_views(out, "nyuv2", t, items, NYU_K, "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/"
                           "nyu_depth_v2_labeled.mat + indoor_seg_sup/splits.mat (testNdxs)",
                           "Kinect, NYU toolbox in-painted ('depths'), Eigen crop", "indoor")
    return n


DIODE_K = np.array([[886.81, 0, 512], [0, 927.06, 384], [0, 0, 1]])


def diode(raw: Path, out: Path, workers: int):
    base = raw / "diode" / "val"
    jobs = [(d, out) for d in sorted(base.glob("*/scene_*"))]
    return _run(_diode_scene, jobs, workers)


def _diode_scene(d: Path, out: Path):
    def dep(p):
        z = np.load(p.with_name(p.stem + "_depth.npy"))[..., 0]
        return np.where(np.load(p.with_name(p.stem + "_depth_mask.npy")) > 0, z, 0)
    items = [(p.stem, lambda p=p: rgb_of(p), lambda p=p: dep(p)) for p in sorted(d.glob("scan_*/*.png"))]
    return _single_views(out, "diode", f"{d.parent.name}_{d.name}", items, DIODE_K,
                         "http://diode-dataset.s3.amazonaws.com/val.tar.gz", "FARO laser scan (depth_mask applied)",
                         d.parent.name)


# ----------------------------------------------------------------------------------------------------- Sintel
def sintel(raw: Path, out: Path, workers: int):
    base = raw / "sintel"
    jobs = [(d.name, base, out) for d in sorted((base / "training" / "final").iterdir())]
    return _run(_sintel_scene, jobs, workers)


def _sintel_scene(seq: str, base: Path, out: Path):
    """.dpt: float32 z-depth; .cam: M (3x3 intrinsics), N (3x4 camera-from-world).  Arbitrary units -> rescaled to
    a median depth of 2 (metric false)."""
    def rd(p, tag):
        with open(p, "rb") as f:
            assert np.fromfile(f, np.float32, 1)[0] == 202021.25, p
            if tag == "dpt":
                w, h = np.fromfile(f, np.int32, 2)
                return np.fromfile(f, np.float32, w * h).reshape(h, w)
            return np.fromfile(f, np.float64, 9).reshape(3, 3), np.fromfile(f, np.float64, 12).reshape(3, 4)
    imgs = sorted((base / "training" / "final" / seq).glob("frame_*.png"))
    sel = stride(len(imgs))
    deps = [rd(base / "training" / "depth" / seq / (imgs[i].stem + ".dpt"), "dpt") for i in sel]
    sw = writer(out, "sintel", seq, "http://sintel.is.tue.mpg.de (MPI-Sintel-training_images.zip, final pass; "
                "MPI-Sintel-depth-training-20150305.zip)", "rendered", metric=False, synthetic=True, dynamic=True,
                kind="movie", units=units_for(deps))
    q = sw.sequence("final")
    for i, dep in zip(sel, deps):
        M, N = rd(base / "training" / "camdata_left" / seq / (imgs[i].stem + ".cam"), "cam")
        q.add(imgs[i].stem, rgb_of(imgs[i]), dep, M, np.vstack([N, [0, 0, 0, 1]]))
    return sw.close()


# ----------------------------------------------------------------------------------------------------- TUM RGB-D
TUM_K = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]])   # fr3 (images are not distorted)


def _tum_list(p: Path):
    rows = [ln.split() for ln in p.read_text().splitlines() if ln and not ln.startswith("#")]
    return np.array([float(r[0]) for r in rows]), rows


def _tum_like(d: Path, q, K, pose_fn, undist=None):
    """TUM-format recording: colour frames matched to the nearest depth frame and the nearest mocap pose (both
    <= 20 ms), at most MAX_FRAMES frames; depth / 5000 (registered to colour)."""
    from scipy.spatial.transform import Rotation
    tr, rr = _tum_list(d / "rgb.txt")
    td, rd = _tum_list(d / "depth.txt")
    tg, rg = _tum_list(d / "groundtruth.txt")
    keep = []
    for k, t in enumerate(tr):
        jd, jg = np.abs(td - t).argmin(), np.abs(tg - t).argmin()
        if abs(td[jd] - t) <= 0.02 and abs(tg[jg] - t) <= 0.02:
            keep.append((k, jd, jg))
    for i in stride(len(keep)):
        k, jd, jg = keep[i]
        g = np.array(rg[jg][1:8], np.float64)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(g[3:]).as_matrix()
        T[:3, 3] = g[:3]
        rgb = rgb_of(d / rr[k][1])
        dep = cv2.imread(str(d / rd[jd][1]), cv2.IMREAD_UNCHANGED) / 5000.0
        if undist is not None:
            rgb = cv2.remap(rgb, *undist, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            dep = cv2.remap(dep.astype(np.float32), *undist, cv2.INTER_NEAREST)
        q.add(f"{tr[k]:.6f}", rgb, dep, K, inv(pose_fn(T)), t=tr[k] - tr[0])


BONN_K = np.array([[542.822841, 0, 315.593520], [0, 542.576870, 237.756098], [0, 0, 1]])
BONN_DIST = np.array([0.039903, -0.099343, -0.000730, -0.000144, 0.0])
T_ROS = np.array([[-1.0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
T_M = np.array([[1.0157, 0.1828, -0.2389, 0.0113], [0.0009, -0.8431, -0.6413, -0.0098],
                [-0.3009, 0.6147, -0.8085, 0.0111], [0, 0, 0, 1]])   # sensor -> markers (Bonn calibration)


def bonn(raw: Path, out: Path, workers: int):
    """Bonn RGB-D Dynamic (balloon2, crowd2, crowd3, person_tracking2, synchronous: the MonST3R / CUT3R set): one
    scene (one room, one mocap frame), each recording a session.  Camera pose = T_ROS^-1 T_gt T_ROS T_m (dataset
    page; T_m's rotation block, not quite orthonormal in the published calibration, is projected onto SO(3));
    colour and registered depth undistorted with the published radial-tangential coefficients."""
    U, _, Vt = np.linalg.svd(T_M[:3, :3])
    tm = T_M.copy()
    tm[:3, :3] = U @ Vt
    undist = cv2.initUndistortRectifyMap(BONN_K, BONN_DIST, None, BONN_K, (640, 480), cv2.CV_32FC1)
    sw = writer(out, "bonn_dynamic", "bonn", "https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/",
                "Xtion, registered to colour by the driver (/5000), undistorted", metric=True, dynamic=True,
                kind="indoor")
    for d in sorted(p for p in (raw / "bonn").glob("rgbd_bonn_*") if (p / "rgb.txt").is_file()):
        _tum_like(d, sw.sequence(d.name[len("rgbd_bonn_"):]), BONN_K,
                  lambda T: np.linalg.inv(T_ROS) @ T @ T_ROS @ tm, undist)
    return sw.close()


def tum(raw: Path, out: Path, workers: int):
    """fr3 walking_{xyz,static,halfsphere,rpy}: one scene (one room, one mocap frame), each recording a session;
    groundtruth.txt = colour camera-to-world (OpenCV axes)."""
    sw = writer(out, "tum_dynamic", "fr3_walking", "https://cvg.cit.tum.de/data/datasets/rgbd-dataset (fr3 walking_*)",
                "Kinect, registered to colour by the driver (/5000)", metric=True, dynamic=True, kind="indoor")
    for d in sorted(p for p in (raw / "tum").glob("rgbd_dataset_freiburg3_walking_*") if (p / "rgb.txt").is_file()):
        _tum_like(d, sw.sequence(d.name.split("_", 3)[-1]), TUM_K, lambda T: T)
    return sw.close()


# ----------------------------------------------------------------------------------------------------- driver
def _run(fn, jobs, workers):
    with ProcessPoolExecutor(workers) as ex:
        return sum(ex.map(fn, *zip(*jobs)))


DATASETS = {"eth3d": eth3d, "7scenes": sevenscenes, "nrgbd": nrgbd, "dtu": dtu, "hiroom": hiroom, "nyuv2": nyuv2,
            "diode": diode, "sintel": sintel, "tum_dynamic": tum, "bonn_dynamic": bonn}


def main():
    global MAX_FRAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(DATASETS))
    ap.add_argument("raw", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    MAX_FRAMES = a.max_frames
    n = DATASETS[a.dataset](a.raw, a.out, a.workers)
    names = write_index(a.out / a.dataset)
    print(f"{a.dataset}: {len(names)} scenes, {n} frames")


if __name__ == "__main__":
    main()
