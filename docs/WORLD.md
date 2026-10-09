# CROSS World: a 3D reconstruction of a CROSS map

`cross_world/` turns a saved CROSS map into a 3D Gaussian-splatting scene that renders from any viewpoint (novel view
synthesis), and exports it to a web viewer. A CROSS map is a topological graph whose permanent nodes are posed keyframe
images; that is all a radiance field needs: posed images, and depth to start from.

```
map.pkl ──► posed keyframes ──► depth ──► chunks ──► per chunk: Gaussians from depth, gsplat training ──► world.pt
(+ source frames, + captured frames)        (sensor / stereo)                                         └──► SPZ + viewer
```

## Install

The reconstruction needs [gsplat](https://github.com/nerfstudio-project/gsplat) (CUDA rasterizer, Apache-2.0) on top of
the CROSS dependencies; `lpips` only for the evaluation:

```bash
uv pip install --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3"   # needs nvcc (CUDA_HOME)
uv pip install lpips plyfile
```

On a new GPU generation set `TORCH_CUDA_ARCH_LIST` (e.g. `12.0` for an RTX 5090) before building.

## Run

```bash
# 1. a map (any mode), optionally keeping extra full-resolution frames for the reconstruction (section 3)
python scripts/map_and_reloc.py --map $SEQ --query $SEQ --out runs/k07 --config configs/outdoor.yaml --skip-reloc \
    --set mapping.world_capture.enabled=true

# 2. the reconstruction: train, evaluate on held-out keyframes (+ source frames between keyframes)
python -m cross_world.cli build --map runs/k07/map.pkl --source $SEQ --capture --out worlds/k07 --novel 150

# 3. web viewer data (SPZ splats, keyframe graph, thumbnails); serve cross_world/viewer/index.html next to it
python -m cross_world.cli export --world worlds/k07/world.pt --map runs/k07/map.pkl --source $SEQ --out web/k07

# 4. after the map changed (loop closure, merged session): move the reconstruction with it, no retraining
python -m cross_world.cli repose --world worlds/k07/world.pt --map runs/k07_v2/map.pkl --out worlds/k07/world_v2.pt
```

`--source` is the prepared sequence folder of the mapping session (`calib.json`, `left/` or `rgb/`, `right/`, `depth/`,
`times.txt`, `poses_left.txt`): the keyframes then use their source frames at full resolution, uncropped, instead of the
images the map stores (<= 512 px; centre-cropped to an aspect ratio of at least 1:2 in the stereo mode). Without it the
stored images and the intrinsics saved with the map are used (maps saved before the `camera` field need `--source`).

Settings: `--set field=value` for `BuildConfig` / `TrainConfig` (`cross_world/world.py`, `cross_world/gaussians.py`),
e.g. `--set max_views=150 train.steps_per_view=80 train.sh_degree=2`. `--chunks 0,3` trains a subset of the chunks
(several GPUs / servers in parallel).

## How it works

- **Views.** Permanent keyframes with images (temporary keyframes have none), posed by hypothesis 0's component of
  the map (camera-to-map, OpenCV axes). Every 8th keyframe is held out for evaluation.
- **Depth.** Sensor depth (RGB-D), or semi-global matching of the stored stereo pair with a left-right check (stereo).
  Monocular maps have no depth source yet (the Gaussians would start from the other views' depth only).
- **Chunks.** Keyframe camera centres on the ground plane are split at the median until a cell holds at most
  `max_views` (200) keyframes. A chunk trains on the keyframes in its cell plus a margin, and on any keyframe that sees
  enough of the cell (VastGaussian-style visibility selection); it keeps the Gaussians in its own cell (near layer) and
  those beyond the mapped region (far layer: sky, distant scenery), which a renderer draws only while the camera is in
  that chunk. Chunks are independent: memory per chunk is bounded and they train in parallel.
- **Training** (gsplat): one Gaussian per voxel of back-projected depth (voxel size grows with depth; coarsened to fit
  70 % of the budget), MCMC densification with a budget per view, L1 + D-SSIM, L1 on inverse depth, per-view pose
  refinement (CROSS keyframe poses carry 0.2-0.6 degree errors, a few pixels), per-view affine colour (exposure).
- **Anchoring.** Every Gaussian is anchored to its four nearest keyframes. `World.repose(new_poses)` moves it with the
  weighted blend of their pose corrections (embedded deformation over the map graph), so a re-optimised or extended
  map does not need a new reconstruction.
- **Export.** SPZ v2 (gzip of 8-bit / 24-bit quantised splats, ~14 bytes per Gaussian with SH degree 1; read by
  [Spark](https://sparkjs.dev)) per chunk and layer, files below 14 MB; standard 3DGS PLY with `--ply`.

## 3. Capturing frames for a later reconstruction (`mapping.world_capture`)

A map keeps few keyframes (KITTI 07: 300 of 1101 frames, one every ~2.3 m) at <= 512 px. With
`--set mapping.world_capture.enabled=true` the mapping run also keeps input frames every `min_translation` m /
`min_rotation_deg` deg (default 0.25 m / 5 deg; 0 / 0 keeps every frame), at the input resolution and uncropped (with the
right image or depth), outside the map's database: `map.pkl.capture/` next to the map (`capture.json` + JPEG frames;
KITTI 07, every frame: 0.38 GB). Each frame's pose is stored relative to its three nearest permanent keyframes, so it
follows later optimisations of the graph. The map itself is unchanged (identical keyframe poses with the option on and
off). `cross_world.cli build --capture` trains on these frames too; evaluation frames are never trained on.

## 4. Viewer

`cross_world/viewer/index.html` loads `scenes.json` (`[{"title", "dir", "description"}]`) and, per scene, the export
directory: splats (Spark 2.2 + three.js 0.180 from jsDelivr), the keyframe frusta, covisibility edges, session path and
chunk cells of the map. Clicking a keyframe flies to its pose; "Compare with photo" overlays the keyframe photo with a
split slider; "Fly the path" moves along the keyframes. Serve it over HTTP (`python -m http.server`).
