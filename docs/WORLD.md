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

The sky model (on by default for stereo maps) needs `transformers` and OneFormer (ADE20K, Swin-T, MIT licence; fetched
from the Hugging Face hub). Its weights are published only as a pickle, which `transformers` loads with torch >= 2.6;
with an older torch convert them once to safetensors and point `CROSS_SKY_MODEL` at the folder:

```bash
uv pip install "transformers>=4.45,<5"
python - <<'PY'   # torch >= 2.6 for the weights_only load; then copy the folder to the training machine
from huggingface_hub import snapshot_download; import torch; from safetensors.torch import save_file
d = snapshot_download("shi-labs/oneformer_ade20k_swin_tiny", local_dir="oneformer", allow_patterns=["*.json", "*.txt", "pytorch_model.bin"])
sd = torch.load(f"{d}/pytorch_model.bin", map_location="cpu", weights_only=True)
save_file({k: v.contiguous() for k, v in sd.items()}, f"{d}/model.safetensors", metadata={"format": "pt"})
PY
export CROSS_SKY_MODEL=$PWD/oneformer
```

Without it the build logs a warning and trains without the sky model.

## Run

```bash
# 1. a map (any mode), optionally keeping the keyframes' full-resolution frames for the reconstruction (section 3)
python scripts/map_and_reloc.py --map $SEQ --query $SEQ --out runs/k07 --config configs/outdoor.yaml --skip-reloc \
    --set mapping.world_capture.enabled=true

# 2. the reconstruction: train, evaluate on held-out keyframes (+ source frames between keyframes, + the right camera)
python -m cross_world.cli build --map runs/k07/map.pkl --source $SEQ --out worlds/k07 --novel 150 --eval-right 60

# 3. web viewer data (SPZ splats, keyframe graph, thumbnails); serve cross_world/viewer/index.html next to it
python -m cross_world.cli export --world worlds/k07/world.pt --map runs/k07/map.pkl --source $SEQ --out web/k07

# 4. after the map changed (loop closure, merged session): move the reconstruction with it, no retraining
python -m cross_world.cli repose --world worlds/k07/world.pt --map runs/k07_v2/map.pkl --out worlds/k07/world_v2.pt
```

`--source` is the prepared sequence folder of the mapping session (`calib.json`, `left/` or `rgb/`, `right/`, `depth/`,
`times.txt`, `poses_left.txt`): the keyframes then use their source frames at full resolution, uncropped, instead of the
images the map stores (<= 512 px; centre-cropped to an aspect ratio of at least 1:2 in the stereo mode). Without it the
stored images and the intrinsics saved with the map are used (maps saved before the `camera` field need `--source`).

Settings: the code defaults (`BuildConfig` / `TrainConfig` in `cross_world/world.py`, `cross_world/gaussians.py`), then
the defaults of the map's mode (`cross_world/configs/<mode>.yaml`; `--no-mode-defaults` skips them), then `--config
file.yaml`, then `--set field=value`, e.g. `--set max_views=150 train.steps_per_view=80 train.sh_degree=2`. `--chunks
0,3` trains a subset of the chunks (several GPUs / servers in parallel).

Evaluation: held-out keyframes (every 8th), `--novel N` source frames between keyframes, `--eval-right N` the right
camera of N of those (stereo: a viewpoint the baseline beside the path, never trained on, where floaters show), all after
a test-time pose alignment; PSNR / SSIM / LPIPS, depth error and `floater_px` (share of the pixels with depth where the
render is > 15 % in front of it) where the view has depth, `--sky-metric` the Gaussians' opacity on sky pixels.

## How it works

- **Views.** Permanent keyframes with images (temporary keyframes have none), posed by hypothesis 0's component of
  the map (camera-to-map, OpenCV axes). Every 8th keyframe is held out for evaluation.
- **Depth** (`--set depth=...`). `auto`: sensor depth (RGB-D), else semi-global matching of the stereo pair with a
  left-right check (`sgbm`). Learned options for stereo maps: `fstereo` (FoundationStereo, the most accurate),
  `vggt` (VGGT-Omega on the pair, scaled by the baseline), `vggt_sgbm` (its scale from SGBM), `fused` (SGBM where it
  matched, scaled VGGT-Omega elsewhere). KITTI 07, 40 keyframes against LiDAR (`python -m cross_world.depth_eval`):

  | depth | AbsRel | delta < 1.25 | LiDAR points covered | s / view (RTX 5090) |
  |---|---|---|---|---|
  | `fstereo` | 0.041 | 0.954 | 97 % | 1.1 |
  | `sgbm` | 0.044 | 0.955 | 53 % | 0.4 |
  | `fused` | 0.054 | 0.945 | 99 % | 0.7 |
  | `vggt_sgbm` | 0.058 | 0.949 | 97 % | 0.7 |
  | `vggt` | 0.119 | 0.879 | 97 % | 0.3 |

  FoundationStereo is not vendored (NVIDIA licence, non-commercial): clone github.com/NVlabs/FoundationStereo, set
  `FOUNDATION_STEREO_REPO` to it, download `23-51-11/model_best_bp2.pth` + `cfg.yaml` (e.g. from the Hugging Face
  mirror `yizhouzhao-nv/FoundationStereo-Backup`) and pass `--set depth=fstereo fstereo_checkpoint=...`; it needs
  `timm omegaconf trimesh joblib open3d pandas scikit-image`. Monocular maps have no depth source yet.
- **Chunks.** Keyframe camera centres on the ground plane are split at the median until a cell holds at most
  `max_views` (200) keyframes. A chunk trains on the keyframes in its cell plus a margin, and on any keyframe that sees
  enough of the cell (VastGaussian-style visibility selection); it keeps the Gaussians in its own cell (near layer) and
  those beyond the mapped region (far layer: sky, distant scenery), which a renderer draws only while the camera is in
  that chunk. Chunks are independent: memory per chunk is bounded and they train in parallel.
- **Training** (gsplat): one Gaussian per voxel of back-projected depth (voxel size grows with depth; coarsened to fit
  70 % of the budget), MCMC densification with a budget per megapixel of training images, L1 + D-SSIM, L1 on inverse
  depth, per-view pose refinement (CROSS keyframe poses carry 0.2-0.6 degree errors, a few pixels) and affine colour
  (exposure), both as sparse embeddings with lazy Adam (only the batch's views move), a penalty on needle-shaped
  Gaussians (largest / middle scale > 10: right edge-on from the training rays, streaks from elsewhere), opacity
  pruning after training.
- **Free-space carving** (`train.carve_every`, default 200 for stereo maps): every 200 steps 16 random training views
  vote with their depth; a Gaussian in front of the observed surface (> 15 %) in two of them and on it in none loses its
  opacity, and MCMC moves it elsewhere. These are the floaters a training view explains (exposure, occlusion edges)
  and every other viewpoint sees in the air. KITTI 07: floater pixels -35 %, +0.1 dB on the path, +0.33 dB 0.54 m
  beside it (right camera); indoors (home1-1, sensor depth) mixed (-0.4 dB held-out, +0.12 dB between keyframes), so it
  is off for RGB-D maps. `cross_world.cli clean --carve` applies it once after training, with every training view. Known issue: indoor chunks with large Gaussian budgets (>= 3.6M; full-resolution 848x480
  views or many captured frames) can collapse (opacities go to zero); 512 px and <= 1.3M Gaussians train reliably
  indoors (`--max-side 512`, `--set train.cap_max=1331420`), outdoor chunks were fine at 4-6M.
- **Sky** (`sky`, default on for stereo maps). The sky has no depth and the camera sees only a narrow band of it
  (KITTI: 29 degrees vertically); painted by Gaussians at arbitrary distances, it floats as white and dark patches once
  the viewpoint leaves the path. As in street-scene splatting (Street Gaussians, OmniRe, PVG), OneFormer masks the sky
  in the training views, an equirectangular sky texture is composited behind the Gaussians, and a cross-entropy on the
  accumulated opacity keeps the Gaussians off the sky pixels (and on the others); sky pixels give no depth. Texels no
  view saw (the zenith) are filled from the seen ones; the export bakes the texture into a shell of splats in the far
  layer. KITTI 07: Gaussians' opacity on sky pixels 1.00 -> 0.11-0.16, PSNR -0.03 / -0.19 / -0.11 dB (held-out /
  between keyframes / right camera) with better SSIM and LPIPS. A chunk whose views show (almost) no sky trains without
  it.
- **Captured frames** (`--capture`, frames between keyframes kept with `mapping.world_capture.between`) train each
  chunk in a second stage: after the keyframes alone, every captured frame's pose is aligned photometrically against
  the chunk (frames between keyframes carry odometry errors of up to a few degrees indoors), frames that still do not
  fit are dropped, and training continues on all views.
- **Anchoring.** Every Gaussian is anchored to its four nearest keyframes. `World.repose(new_poses)` moves it with the
  weighted blend of their pose corrections (embedded deformation over the map graph), so a re-optimised or extended
  map does not need a new reconstruction.
- **Export.** SPZ v2 (gzip of 8-bit / 24-bit quantised splats, ~14 bytes per Gaussian with SH degree 1; read by
  [Spark](https://sparkjs.dev)) per chunk and layer, files below 14 MB; standard 3DGS PLY with `--ply`.

## 3. Capturing frames for a later reconstruction (`mapping.world_capture`)

A map keeps few keyframes (KITTI 07: 300 of 1101 frames, one every ~2.3 m) at <= 512 px, centre-cropped in the stereo
mode. With `--set mapping.world_capture.enabled=true` the mapping run also keeps the input frame of every new permanent
keyframe at the input resolution and uncropped (with the right image or depth), outside the map's database:
`map.pkl.capture/` next to the map (`capture.json` + JPEG frames). The map itself is unchanged (identical keyframe poses
with the option on and off). Without `--source`, keyframes then use their captured frames instead of the stored images:
in the stereo mode this is what matters (KITTI 07: +2.4 dB held-out, +3.1 dB between keyframes over the stored 512x256
centre crops).

`mapping.world_capture.between=true` also keeps frames between keyframes every `min_translation` m / `min_rotation_deg`
deg (default 0.25 m / 5 deg; 0 / 0 keeps every frame; KITTI 07, every frame: 0.38 GB), each posed relative to its three
nearest permanent keyframes so that it follows later optimisations of the graph, and `cross_world.cli build --capture`
trains on them (evaluation frames never). Off by default: against keyframes only with the same Gaussian budget and
training steps, KITTI 07 gains 1.2 dB at held-out keyframes (2.3 m from the nearest training view) but only 0.15 dB
between keyframes, where the renders get blurrier (LPIPS 0.262 vs 0.225); home1-1 gains nothing (its frames between
keyframes carry ~0.7 deg pose errors, aligned in a second stage but not perfectly); training takes 3.5x as long.

`python -m cross_world.cli clean --world W --map M --out W2` removes Gaussians fewer than two training views see
(MCMC rarely leaves any after opacity pruning: < 0.2 % on home1-1).

## 4. Viewer

`cross_world/viewer/index.html` loads `scenes.json` (`[{"title", "dir", "description"}]`) and, per scene, the export
directory: splats (Spark 2.2 + three.js 0.180 from jsDelivr), the keyframe frusta, covisibility edges, session path and
chunk cells of the map. Clicking a keyframe flies to its pose; "Compare with photo" overlays the keyframe photo with a
split slider; "Fly the path" moves along the keyframes; "Overview" looks from 35 degrees above and hides what is only
right near the training viewpoints (the export's layers: `above` = higher than the cameras plus a margin (ceiling,
sky floaters), `offview` = needles and large faint Gaussians, `far` = beyond the mapped region). Serve it over HTTP
(`python -m http.server`); a host that serves only web file types can take the binary files as base64 text
(`world.json` `"base64": true`, files `*.b64.txt`).
