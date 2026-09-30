# SimChange-Rearrange: HSSD scenes with quantified object rearrangement

Date: 2026-09-06

## Motivation

The three original SimChange scenes have 6 (classroom), 9 (apartment) and 24 (Lone Monk) movable objects, displaced by at
most 1.5 m; every `move_*` variant gave RS = 1.00 for every system. To measure tolerance to object rearrangement the
benchmark needs environments whose appearance is dominated by many independently placed objects.

## Scenes

Source: [HSSD](https://huggingface.co/datasets/hssd/hssd-hab) (Habitat Synthetic Scenes Dataset, CC BY-NC 4.0): 211
artist-authored interiors, every object a separate model with a semantic category. Two single-floor scenes were chosen
after inspecting the four largest object-rich candidates (`scripts/sim/hssd_download.py` lists the statistics):

| key | HSSD id | footprint | placed objects / distinct models | rooms | rearrangement pool |
|---|---|---|---|---|---|
| `hssd_house` | 104348010_171512832 | 41 x 24 m | 708 / 383 | 22 (kitchen, dining, two living rooms, five bedrooms, office, bathrooms, garage, terrace) | 281 |
| `hssd_restaurant` | 103997718_171030855 | 49 x 44 m | 1410 / 703 | dining hall, games corner, corridor, lounge, bar, kitchen, terrace | 1012 |

Import (`scripts/sim/hssd_to_blend.py`): stage and object glbs are converted from Habitat's Y-up frame to Blender's Z-up
frame (rotation quaternion `[w, x, y, z]`, `translation_origin: asset_local`); each model is imported once and shared by
its instances; every instance carries `hssd_category`, `hssd_super`, `hssd_index`, a unique `pass_index`, and lamp
objects are tagged. Object render assets come from `hssd-models` (PNG textures); the decomposed parts that only exist in
`hssd-hab` use KTX2/Basis textures and are transcoded with the Khronos `ktx` tool (`scripts/sim/glb_debasis.py`). Door
leaves are hidden (doors open), window glass is made transparent to shadow and diffuse rays (Cycles has no caustics),
a Nishita sky provides sun and daylight through light portals placed at the windows, and interior lamps are created at
the tagged lamp objects for the evening / night presets. A region-based auto-exposure lowers the exposure on the terrace.

## Trajectories

One map loop per scene, planned with the clearance-aware grid planner restricted to the annotated rooms
(`restrict_to_regions`), 0.1 m per station, 1.3 m camera height, 90 degrees HFOV, 640 x 480, left + 0.3 m right camera:

* house: garage, west hallway, kitchen (around the island), terrace, dining, living room, entry, east hallway, utility
  room, bedroom 2, office, bedroom 1, back through the living room, then hallway, kitchen and terrace again in the same
  direction (**loop closure**: stations 1600 to 1953 revisit stations 90 to 450, 36 m same direction, 20 m reversed);
  195 m, 1953 stations.
* restaurant: dining hall east aisle, games corner, service corridor, lounge and communal tables, west aisle, terrace and
  back, kitchen, dining hall west aisle and north side, then the east aisle again (**loop closure**, 24 m same direction,
  54 m reversed); 200 m, 2000 stations.

**Kinematic plausibility.** The grid path has cusps at dead-end reversals and sharp corners where the heading changed by
up to 180 degrees between consecutive 0.1 m stations. In-place rotation stations are inserted so that no frame turns by
more than 10 degrees (`insert_turns`, `max_turn_deg_per_frame`): 352 extra stations in the house (2305 total) and 300 in
the restaurant (2300). The first render pass produced the translation stations; the rotation stations were rendered
separately (`--turn-frames-only`) and spliced in (`scripts/sim/splice_turn_frames.py`); `calib.json` of a spliced
traversal carries `turn_frames_inserted`. Without this, ORB-SLAM3 fragmented the map traversal into a dozen sub-maps
(the same defect exists, less severely, in the Lone Monk loop: 7 jumps above 45 degrees).

## Rearrangement model (`scripts/sim/rearrange_plan.py`, `hssd_classes.py`)

* Mobility classes from the HSSD category: `structural` (never changes: windows, doors, wall-mounted cabinets, fixtures,
  built-in appliances), `heavy` furniture, `light` furniture, `clutter`, `decor` (wall items).
* Pool: non-structural objects in the rooms the map visits (or within 3 m of the path).
* Level L: `round(L * |pool|)` objects are drawn without replacement with class weights clutter 1.0, light 0.8, decor 0.5,
  heavy 0.3.
* Operations per object (class-dependent mix): `relocate` (free spot in the same room for floor objects, on the same
  support surface for supported objects; 20 % of relocations may go to a neighbouring room), `jitter` (0.1 to 0.5 m push,
  up to 20 degrees), `remove`, `swap` (with a same-class object of a different model). Objects standing on a moved or
  removed piece follow it; objects resting on the static stage (counters) can only be pushed, swapped or removed. Placement
  checks use a leg-level occupancy grid (chairs tuck under tables), forbid new overlaps and deeper wall penetration than in
  the original layout, and keep a 0.45 m corridor around the trajectory free.
* Variant names: `rearr_<percent>[_s<seed>][_remove|_relocate|_jitter]`, combinable with the other families via `+`
  (e.g. `rearr_50+light_night`, `rearr_100+light_evening+reverse`).

## Quantification (`scripts/sim/quantify_rearrangement.py`)

Object-index passes rendered along the map trajectory (`--quantify --id-pass`, 320 x 240, every 5th station) for the
map layout and each variant give the *changed-object pixel ratio* (CPR) per frame: the fraction of pixels that belong to
a changed object in either layout or whose object identity differs. Reported: mean, median, 90th percentile and the
ceiling (fraction of pixels on any placed object: 34 % house, 37 % restaurant; on the pool: 16 % / 21 %).

| level | house CPR mean / p90 | restaurant CPR mean / p90 |
|---|---|---|
| 10 % | 0.014 / 0.047 | 0.013 / 0.032 |
| 25 % | 0.037 / 0.108 | 0.055 / 0.105 |
| 50 % | 0.069 / 0.152 (seeds 1, 2: 0.077, 0.088) | 0.096 / 0.168 (seeds: 0.099, 0.121) |
| 75 % | 0.116 / 0.217 | 0.159 / 0.283 |
| 100 % | 0.215 / 0.347 | 0.251 / 0.377 |

## Variant set (22 traversals per scene)

`map`, `rearr_10 rearr_25 rearr_50 rearr_75 rearr_100`, `rearr_50_s1 rearr_50_s2`, `rearr_50_remove rearr_50_relocate`,
`light_morning light_evening light_night light_overcast`, `offset_1.0 yaw_45 reverse half`,
`rearr_50+light_night rearr_50+reverse rearr_100+light_evening+reverse rearr_50+offset_1.0+light_overcast`.

## Pipeline

* Assets and renders: `data/sim/assets/hssd` and `data/sim/hssd_{house,restaurant}` (see the README for the commands).
* Rendering: `scripts/sim/gen_simchange.py --scene hssd_<id>` (Blender `bpy` venv), one traversal per variant; the
  rotation frames are spliced in with `scripts/sim/splice_turn_frames.py`.
* Evaluation: `scripts/run_sim_experiments.sh <scene> [systems...]` (CROSS-stereo, CROSS-PnP GT / SGBM, ORB-SLAM3 stereo,
  RTAB-Map RGB-D / stereo, MASt3R-SLAM subset). Protocol: 100-frame trials, stride 50, r_D = 2 m, map = the loop-closing
  `map` traversal.
* Results: `scripts/make_hssd_figures.py`
  (tables `report/tables/hssd_*`, figures `report/figures/hssd_*`, `outputs/summary_hssd.md`).
