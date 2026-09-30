#!/usr/bin/env python3
"""Self-contained HTML check package for the new HSSD rearrangement scenes (images embedded as JPEG data URIs).

    .venv/bin/python scripts/sim/build_scene_check_report.py --work temp/hssd --out outputs/hssd_scene_check/index.html [--fragment body.html]
"""
import argparse, base64, io, json
from pathlib import Path
import numpy as np
import cv2


def img_uri(path, max_w=1500, q=82):
    im = cv2.imread(str(path))
    if im is None:
        return None
    h, w = im.shape[:2]
    if w > max_w:
        im = cv2.resize(im, None, fx=max_w / w, fy=max_w / w, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def fig(path, caption, max_w=1500):
    uri = img_uri(path, max_w)
    if uri is None:
        return f'<p class="missing">missing figure: {path}</p>'
    return f'<figure><img src="{uri}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


def plan_row(p):
    ops = p["changed_by_op"]
    n_rm = ops.get("remove", 0) + ops.get("follow_remove", 0)
    n_mv = ops.get("relocate", 0) + ops.get("jitter", 0) + ops.get("swap", 0) + ops.get("follow", 0)
    return (f"<tr><td>{int(round(p['level']*100))} %</td><td>{p['seed']}</td><td>{p['n_changed']} / {p['pool_size']}</td>"
            f"<td>{p['changed_by_class'].get('clutter',0)} / {p['changed_by_class'].get('light',0)} / {p['changed_by_class'].get('heavy',0)} / {p['changed_by_class'].get('decor',0)}</td>"
            f"<td>{ops.get('relocate',0)} / {ops.get('jitter',0)} / {ops.get('swap',0)} / {ops.get('follow',0)} / {n_rm}</td>"
            f"<td>{p['mean_displacement']:.2f} m</td></tr>")


def quant_rows(q):
    rows = []
    for v, r in q.items():
        rows.append(f"<tr><td><code>{v}</code></td><td>{int(round(r['level']*100))} %</td><td>{r['seed']}</td><td class='num'>{r['n_changed']}</td>"
                    f"<td class='num'>{r['cpr_obj_mean']:.3f}</td><td class='num'>{r['cpr_obj_median']:.3f}</td><td class='num'>{r['cpr_obj_p90']:.3f}</td><td class='num'>{r['cpr_strict_mean']:.3f}</td>"
                    f"<td class='num'>{100*r['frac_frames_cpr_obj_gt_0.2']:.0f} %</td><td class='num'>{r['visible_changed_objects_mean']:.1f}</td></tr>")
    return "\n".join(rows)


def scene_section(name, title, w, layout_json, path_npy, quant_json, plan_dir, figs, desc, eyebrow=""):
    lay = json.loads(Path(layout_json).read_text())
    P = np.load(path_npy)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hssd_classes import mobility_class
    cls = {}
    for o in lay["objects"]:
        c = mobility_class(o["category"], o.get("maxdim", -1), o.get("super", ""), o["bbox_min"][2]); cls[c] = cls.get(c, 0) + 1
    (x0, y0, _), (x1, y1, _) = lay["stage_bbox"]
    q = json.loads(Path(quant_json).read_text()) if Path(quant_json).is_file() else {}
    plans = []
    for v in ("rearr_25", "rearr_50", "rearr_100"):
        pj = Path(plan_dir) / v / "plan.json"
        if pj.is_file():
            plans.append(json.loads(pj.read_text()))
    pool = plans[0]["pool_size"] if plans else "?"
    visited = plans[0]["visited_rooms"] if plans else []
    any_q = next(iter(q.values())) if q else None
    obj_px = f"{100*any_q['obj_ratio_mean']:.0f} %" if any_q else "?"
    mov_px = f"{100*any_q['movable_ratio_mean']:.0f} %" if any_q else "?"
    html = f"""
<section id="{name}">
<h2><span class="eyebrow">{eyebrow}</span>{title}</h2>
<p>{desc}</p>
<div class="tw"><table class="kv">
<tr><th>HSSD scene id</th><td><code>{lay['scene']}</code></td><th>stage footprint</th><td>{x1-x0:.0f} x {y1-y0:.0f} m, single floor</td></tr>
<tr><th>placed objects</th><td>{len(lay['objects'])} ({len(set(o['template'] for o in lay['objects']))} distinct models)</td>
<th>mobility classes</th><td>structural {cls.get('structural',0)}, heavy {cls.get('heavy',0)}, light {cls.get('light',0)}, clutter {cls.get('clutter',0)}, decor {cls.get('decor',0)}</td></tr>
<tr><th>map trajectory</th><td>{len(P)} stations, {0.1*len(P):.0f} m (0.1 m step, 1.3 m camera height, 90&deg; HFOV, 640x480)</td>
<th>rooms visited</th><td>{len(visited)}: {", ".join(visited)}</td></tr>
<tr><th>rearrangement pool</th><td>{pool} objects in visited rooms (or within 3 m of the path)</td>
<th>pixels on objects along the map</th><td>{obj_px} on any placed object, {mov_px} on pool objects (mean over map frames)</td></tr>
</table></div>
{fig(figs['topdown'], 'Top-down view of the imported scene (Workbench render, ceilings clipped at 2.3 m).')}
{fig(figs['layout'], 'Object footprints coloured by mobility class, room regions (dashed), occupancy (dark) and the final map trajectory with station numbers (green dot = start, orange square = end). The rearrangement figures below were computed on the trajectory before the loop-closure extension was appended; pools differ by a few objects.')}
{fig(figs['light'], 'Lighting presets at four stations along the trajectory (rows: map = default afternoon sun, morning, evening, night, overcast). Evening and night switch on the interior lamps; the terrace frames use the region-based auto-exposure.')}
<h3>Rearrangement levels</h3>
{fig(figs['plan50'], 'Rearrangement plan at level 50 % (seed 0): black arrows = relocations, green = small pushes (jitter), pink = swaps, grey = objects following a moved support, x = removed objects.')}
<div class="tw"><table>
<tr><th>level</th><th>seed</th><th>changed / pool</th><th>by class<br>clutter / light / heavy / decor</th><th>by operation<br>relocate / jitter / swap / follow / removed</th><th>mean displacement</th></tr>
{''.join(plan_row(p) for p in plans)}
</table></div>
{fig(figs['pairs0'], 'Same viewpoint under the map layout and the 25 / 50 / 100 % rearrangements (columns), stations along the trajectory (rows).')}
{fig(figs['pairs1'], 'Same viewpoint under the map layout and the 25 / 50 / 100 % rearrangements, further stations.')}
<h3>Measured visual change</h3>
{fig(figs['quant'], 'Changed-object pixel ratio (CPR) along the map trajectory for each level (left) and its mean / 90th percentile versus level, with other random seeds and single-operation variants (right). The dashed line is the pixel fraction of all placed objects, the dotted line that of the rearrangement pool: they bound what any rearrangement can change.')}
<div class="tw"><table>
<tr><th>variant</th><th>level</th><th>seed</th><th class="num">objects changed</th><th class="num">CPR mean</th><th class="num">CPR median</th><th class="num">CPR p90</th><th class="num">strict change mean</th><th class="num">frames with CPR &gt; 0.2</th><th class="num">changed objects visible / frame</th></tr>
{quant_rows(q)}
</table></div>
</section>
"""
    return html


CSS = """
:root{--bg:#f3f3ee;--ink:#1b201e;--muted:#5b625e;--rule:#d9dad3;--accent:#1f5f6b;--accent-soft:#dfe9ea;--warn-bg:#fbf4e3;--warn-rule:#c9a04a;--code-bg:#e8e9e2;--th-bg:#e4e8e5;--fig-bg:#ffffff}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#151918;--ink:#e5e7e2;--muted:#9ba39e;--rule:#2b3230;--accent:#7cc0cb;--accent-soft:#1d2f32;--warn-bg:#2a2619;--warn-rule:#a7873a;--code-bg:#232826;--th-bg:#1f2624;--fig-bg:#f4f4f0}}
:root[data-theme="dark"]{--bg:#151918;--ink:#e5e7e2;--muted:#9ba39e;--rule:#2b3230;--accent:#7cc0cb;--accent-soft:#1d2f32;--warn-bg:#2a2619;--warn-rule:#a7873a;--code-bg:#232826;--th-bg:#1f2624;--fig-bg:#f4f4f0}
html{background:var(--bg)}
body{font-family:"Source Sans 3","Source Sans Pro","Helvetica Neue",Arial,sans-serif;font-size:17px;line-height:1.5;color:var(--ink);background:var(--bg);margin:0;padding:32px 24px 80px}
main{max-width:1180px;margin:0 auto}
h1,h2,h3{font-family:Archivo,"Helvetica Neue",Arial,sans-serif;font-weight:600;letter-spacing:-0.01em;text-wrap:balance;color:var(--ink)}
h1{font-size:2.1rem;line-height:1.15;margin:0 0 .4rem}
h2{font-size:1.45rem;margin:3rem 0 .8rem;padding-top:1rem;border-top:2px solid var(--rule)}
h3{font-size:1.1rem;margin:2rem 0 .6rem}
.eyebrow{font-family:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);display:block;margin-bottom:.25rem}
.lede{font-size:1.1rem;color:var(--muted);max-width:72ch;margin:0 0 1.5rem}
p,li{max-width:76ch}
.note{background:var(--warn-bg);border-left:4px solid var(--warn-rule);padding:10px 14px;max-width:none}
.reco{background:var(--accent-soft);border-left:4px solid var(--accent);padding:12px 16px;margin:1.2rem 0}
.reco p{max-width:none;margin:.3rem 0}
table{border-collapse:collapse;margin:1rem 0;font-size:.93rem;font-variant-numeric:tabular-nums;width:100%}
.tw{overflow-x:auto}
th,td{border:1px solid var(--rule);padding:5px 9px;text-align:left;vertical-align:top}
th{background:var(--th-bg);font-family:Archivo,"Helvetica Neue",Arial,sans-serif;font-weight:600;font-size:.86rem}
td.num,th.num{text-align:right}
table.kv th{width:13%;white-space:nowrap} table.kv td{width:37%}
figure{margin:1.4rem 0} figure img{max-width:100%;height:auto;border:1px solid var(--rule);background:var(--fig-bg);display:block}
figcaption{font-size:.9rem;color:var(--muted);margin-top:.4rem;max-width:90ch}
code,kbd{font-family:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.86em;background:var(--code-bg);padding:1px 5px;border-radius:3px}
pre{font-family:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.84rem;background:var(--code-bg);padding:12px 14px;overflow-x:auto;border-radius:4px;line-height:1.45}
pre code{background:none;padding:0}
ul.tight{padding-left:1.2rem} ul.tight li{margin:.25rem 0}
.missing{color:#b0322a}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px 28px}
a{color:var(--accent)}
"""

FONTS = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap">'



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="temp/hssd")
    ap.add_argument("--out", default="outputs/hssd_scene_check/index.html")
    ap.add_argument("--fragment", default=None)
    a = ap.parse_args()
    W = Path(a.work)
    house = scene_section(
        "house", "House 104348010 (41 x 24 m, 708 objects)", W, W / "hssd_104348010.blend.layout.json", W / "path_xy_final.npy",
        "outputs/hssd_quant/104348010/quantification.json", W / "pr",
        {"topdown": W / "hssd_104348010.blend.topdown_small.jpg", "layout": W / "layout_104348010_final.png", "light": W / "light_grid3.jpg",
         "plan50": W / "layout_104348010_plan50.png", "pairs0": W / "pairs_grid/pairs_0.jpg", "pairs1": W / "pairs_grid/pairs_5.jpg",
         "quant": "outputs/hssd_quant/104348010/quantification.png"},
        "A fully furnished single-family house: kitchen with island, dining room, two living rooms, entry, five bedrooms, office, "
        "bathrooms, laundry, garage with cars and tools, and a furnished terrace with a pool. The 167 m loop starts in the garage, "
        "follows the west hallway into the kitchen, circles the island, steps onto the terrace, returns through the dining and living "
        "rooms to the entry, runs down the east hallway with out-and-back visits to the utility room, bedroom 2, the office and "
        "bedroom 1, returns through the living room and closes the loop by repeating the hallway, kitchen and terrace segment in the same "
        "direction (stations 1600 to 1953 revisit stations 90 to 450).", eyebrow="Scene A")
    rest = scene_section(
        "restaurant", "Waterfront restaurant 103997718 (49 x 44 m, 1410 objects)", W, W / "hssd_103997718.blend.layout.json", W / "rpath_xy_final.npy",
        "outputs/hssd_quant/103997718/quantification.json", W / "rp",
        {"topdown": W / "hssd_103997718.blend.topdown_small.jpg", "layout": W / "layout_103997718_final.png", "light": W / "r_light_grid.jpg",
         "plan50": W / "layout_103997718_plan50.png", "pairs0": W / "r_pairs_grid/pairs_0.jpg", "pairs1": W / "r_pairs_grid/pairs_6.jpg",
         "quant": "outputs/hssd_quant/103997718/quantification.png"},
        "An open-plan restaurant and bar on the water: a dining hall with table rows and booths, a games corner, a service corridor "
        "with toilets, a lounge with sofas and communal tables, a bar, a kitchen and a terrace with umbrellas. Most of what the camera "
        "sees is furniture and tableware rather than walls. The 183 m loop goes down the dining hall, through the games corner and the "
        "corridor into the lounge, along the lounge's west aisle onto the terrace and back, through the kitchen, up the dining hall's west "
        "aisle and along its north side to the start, then repeats the east aisle down to the corridor as the loop closure. "
        "(The window panes of this asset are opaque dark glass in HSSD; they stay black in every lighting condition.)", eyebrow="Scene B")
    qa = json.loads(Path("outputs/hssd_quant/104348010/quantification.json").read_text())
    qb = json.loads(Path("outputs/hssd_quant/103997718/quantification.json").read_text())
    def g(q, v, k, f="{:.2f}"):
        return f.format(q[v][k]) if v in q else "?"
    glance = f"""
<div class="tw"><table>
<tr><th></th><th>Scene A: house 104348010</th><th>Scene B: restaurant 103997718</th><th>Lone Monk (current long scene)</th></tr>
<tr><td>footprint</td><td>41 x 24 m, 22 rooms, single floor</td><td>49 x 44 m, open plan + 16 regions</td><td>40 x 40 m cloister</td></tr>
<tr><td>placed objects / distinct models</td><td class="num">708 / 383</td><td class="num">1410 / 703</td><td class="num">24 movable</td></tr>
<tr><td>rearrangement pool (objects that can change)</td><td class="num">281</td><td class="num">1012</td><td class="num">24</td></tr>
<tr><td>map trajectory (with same-direction loop closure)</td><td class="num">195 m, 1953 stations (36 m revisited in the same direction, 20 m reversed)</td><td class="num">200 m, 2000 stations (24 m same direction, 54 m reversed)</td><td class="num">126 m + 15 m replay</td></tr>
<tr><td>pixels on placed objects / on the pool (mean over the map)</td><td class="num">{100*qa['rearr_50']['obj_ratio_mean']:.0f} % / {100*qa['rearr_50']['movable_ratio_mean']:.0f} %</td><td class="num">{100*qb['rearr_50']['obj_ratio_mean']:.0f} % / {100*qb['rearr_50']['movable_ratio_mean']:.0f} %</td><td>n/a</td></tr>
<tr><td>changed-object pixel ratio, level 50 % (mean / p90)</td><td class="num">{g(qa,'rearr_50','cpr_obj_mean')} / {g(qa,'rearr_50','cpr_obj_p90')}</td><td class="num">{g(qb,'rearr_50','cpr_obj_mean')} / {g(qb,'rearr_50','cpr_obj_p90')}</td><td>n/a</td></tr>
<tr><td>changed-object pixel ratio, level 100 % (mean / p90)</td><td class="num">{g(qa,'rearr_100','cpr_obj_mean')} / {g(qa,'rearr_100','cpr_obj_p90')}</td><td class="num">{g(qb,'rearr_100','cpr_obj_mean')} / {g(qb,'rearr_100','cpr_obj_p90')}</td><td>n/a</td></tr>
<tr><td>spread over seeds at 50 % (CPR mean, seeds 0 / 1 / 2)</td><td class="num">{g(qa,'rearr_50','cpr_obj_mean')} / {g(qa,'rearr_50_s1','cpr_obj_mean')} / {g(qa,'rearr_50_s2','cpr_obj_mean')}</td><td class="num">{g(qb,'rearr_50','cpr_obj_mean')} / {g(qb,'rearr_50_s1','cpr_obj_mean')} / {g(qb,'rearr_50_s2','cpr_obj_mean')}</td><td>n/a</td></tr>
<tr><td>render cost per traversal (left + right, one A100)</td><td class="num">about 70 min</td><td class="num">about 80 min</td><td class="num">about 45 min</td></tr>
</table></div>"""
    body = f"""
<header>
<span class="eyebrow">SimChange &middot; new scene check</span>
<h1>Rearrangement scenes from HSSD</h1>
<p class="lede">Two large, object-rich interiors imported from the Habitat Synthetic Scenes Dataset, each with a long map loop, the existing
lighting / viewpoint / direction perturbations, and a new quantified object-rearrangement axis. This package is for verifying the scene before
any traversal set is rendered or any method is run.</p>
</header>
<h2>At a glance</h2>
{glance}
<div class="reco">
<p><b>Recommendation.</b> Use the restaurant (Scene B) as the heavy-rearrangement scene: its pool is 3.6 times larger, the geometry near the camera
is furniture rather than walls, its identical chairs and tables add place aliasing, and the loop is the longest of all scenes. The house (Scene A)
is the more typical service-robot setting (rooms, corridors, doorways) and is ready to render as a second scene if the budget allows; both take
about half a day on the two A100s for the proposed 21 variants.</p>
<p>In both scenes the extreme level changes about a quarter of all pixels on average (peaks near 0.7): walls, floors and ceilings bound what any
rearrangement can do, and that bound (34 to 37 percent of pixels) is reported next to every curve.</p>
</div>

<h2>1. Why these scenes</h2>
<p>The three existing SimChange scenes have 6 (classroom), 9 (apartment) and 24 (Lone Monk) movable objects, displaced by at most 1.5 m, and every
<code>move_*</code> column of the current results is 1.00 for every method. A scene whose appearance is dominated by many independently placed objects
was needed. Candidates:</p>
<table>
<tr><th>source</th><th>scene</th><th>size</th><th>separately placed objects</th><th>verdict</th></tr>
<tr><td>Blender demo files</td><td>Lone Monk, Classroom, Archiviz, Loft, Flat, splash scenes</td><td>one room to 40 x 40 m</td><td>6 to 24 movable props, the rest is baked or structural</td><td>already used / too few objects</td></tr>
<tr><td>HSSD (Habitat Synthetic Scenes Dataset, CC BY-NC 4.0)</td><td>211 artist-authored interiors, 18k object models</td><td>houses 10 x 10 to 60 x 40 m</td><td>every object is its own asset with a semantic category</td><td><b>chosen</b>: Blender-importable after texture conversion, rearrangement-ready by design</td></tr>
<tr><td>HSSD 103997940</td><td>estate with 3 buildings</td><td>57 x 36 m</td><td>639</td><td>rejected: disconnected buildings, sparse rooms</td></tr>
<tr><td>HSSD 106366323</td><td>house with storage</td><td>40 x 39 m</td><td>761 (380 are modular shelf parts)</td><td>rejected: half of the objects are one shelving system</td></tr>
<tr><td>HSSD 104348010</td><td>furnished house</td><td>41 x 24 m</td><td>708 (383 models)</td><td><b>Scene A</b></td></tr>
<tr><td>HSSD 103997718</td><td>waterfront restaurant</td><td>49 x 44 m</td><td>1410 (703 models)</td><td><b>Scene B</b></td></tr>
</table>
<p>Both scenes were imported into Blender 5 (Habitat Y-up to Z-up, one shared mesh per model, semantic tags on every instance, door
leaves opened, glass made light-transmissive, Nishita sky with light portals at the windows, interior lamps at the lamp objects), the free
space was mapped with the existing occupancy tool, and the trajectory was planned with the existing clearance-aware planner restricted to the
annotated rooms.</p>

<h2>2. Rearrangement model and its quantification</h2>
<ul class="tight">
<li><b>Mobility classes</b> from the HSSD category of every object: <em>structural</em> (never changes: windows, doors, wall-mounted cabinets, fixtures,
built-in appliances), <em>heavy</em> furniture (beds, sofas, wardrobes, tables), <em>light</em> furniture (chairs, stools, plants, lamps, bins, boxes),
<em>clutter</em> (dishes, books, cushions, toiletries), <em>decor</em> (pictures, wall clocks).</li>
<li><b>Level L</b> = fraction of the rearrangement pool that changes (pool = non-structural objects in the rooms the map visits). Objects are drawn
without replacement with class weights clutter 1.0, light 0.8, decor 0.5, heavy 0.3, so low levels move what moves in real life; at 100 % everything
including the heavy furniture changes.</li>
<li><b>Operations</b> per changed object: <em>relocate</em> to a free spot in the same room (floor objects) or on the same support surface (objects on
tables and shelves), <em>jitter</em> (0.1 to 0.5 m push and up to 20&deg; turn), <em>remove</em>, <em>swap</em> with another object of the same class and a
different model; objects standing on a moved or removed piece follow it; objects on built-in counters can only be pushed slightly, swapped or removed.
The trajectory corridor stays free; chairs may tuck under tables (leg-level occupancy check).</li>
<li><b>Measured change</b>: object-index passes are rendered along the map trajectory for the map layout and for every variant; the
<em>changed-object pixel ratio</em> (CPR) of a frame is the fraction of its pixels that belong to a changed object in either layout or whose object
identity differs between the two. It is reported per frame (curve), as mean / median / 90th percentile, and against its ceiling, the fraction of pixels
on any placed object.</li>
<li><b>Randomness</b>: seeds 1 and 2 at level 50 % and single-operation variants (only remove / only relocate / only jitter) show the spread.</li>
</ul>
{house}
{rest}
<h2>5. Proposed variant set for the chosen scene (for confirmation)</h2>
<table>
<tr><th>family</th><th>variants</th><th>purpose</th></tr>
<tr><td>rearrangement levels</td><td><code>rearr_10 rearr_25 rearr_50 rearr_75 rearr_100</code> (seed 0)</td><td>tolerance curve versus level</td></tr>
<tr><td>randomness</td><td><code>rearr_50_s1 rearr_50_s2</code></td><td>spread between random rearrangements at the same level</td></tr>
<tr><td>operation ablation</td><td><code>rearr_50_remove rearr_50_relocate</code></td><td>disappearing versus displaced objects</td></tr>
<tr><td>lighting</td><td><code>light_morning light_evening light_night light_overcast</code></td><td>as in the other scenes</td></tr>
<tr><td>viewpoint / direction</td><td><code>offset_1.0 yaw_45 reverse half</code></td><td>as in the other scenes</td></tr>
<tr><td>combined</td><td><code>rearr_50+light_night</code>, <code>rearr_50+reverse</code>, <code>rearr_100+light_evening+reverse</code>, <code>rearr_50+offset_1.0+light_overcast</code></td><td>rearrangement combined with the other perturbations</td></tr>
</table>
<p>Render cost on one A100 (Cycles, 16 samples, left + right at 0.3 m): about 1.3 s per image, i.e. roughly 70 min per 167 m traversal and
80 min per 183 m traversal; the 21 variants above plus the map take about 25 h of GPU time per scene, 12 to 13 h on the two GPUs. Commands
(from the repository root):</p>
<pre><code># one traversal (left + 0.3 m right images, depth, poses); rearrangement plans are generated and stored with the traversal
CUDA_VISIBLE_DEVICES=0 .venv-blender/bin/python scripts/sim/gen_simchange.py --scene hssd_103997718 --out data/sim/hssd_restaurant \
    --occ temp/occ_hssd_103997718.npz --baselines 0.3 --variants map,rearr_50,rearr_50+light_night
# quantify any set of variants from object-index passes (fast, 320x240, every 5th station)
CUDA_VISIBLE_DEVICES=0 .venv-blender/bin/python scripts/sim/gen_simchange.py --scene hssd_103997718 --out temp/hssd_r_quant --quantify --id-pass \
    --frame-stride 5 --width 320 --height 240 --baselines "" --occ temp/occ_hssd_103997718.npz --dump-path --variants map,rearr_10,rearr_25,rearr_50
.venv/bin/python scripts/sim/quantify_rearrangement.py --root temp/hssd_r_quant --variants rearr_10,rearr_25,rearr_50 --out outputs/hssd_quant/103997718</code></pre>
<h2>6. Points to verify</h2>
<ul class="tight">
<li>Scene choice: house (A), restaurant (B), or both? The restaurant has more than twice the objects and a far larger fraction of the view on movable
objects; the house is the more typical service-robot environment with rooms and corridors (stronger place aliasing between similar rooms).</li>
<li>Trajectory: length (167 m / 183 m), rooms covered, out-and-back visits to the bedrooms.</li>
<li>Rearrangement realism: class weights, operation mix (about 30 % of changes are removals at every level), same-room relocation with 20 % cross-room moves.</li>
<li>Levels: 10 / 25 / 50 / 75 / 100 % of the pool, two extra seeds at 50 %, single-operation ablations.</li>
<li>Lighting presets and the auto-exposure on the terrace.</li>
</ul>
"""
    full = f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>Rearrangement Scenes from HSSD</title>{FONTS}<style>{CSS}</style></head><body><main>{body}</main></body></html>"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(full)
    print("wrote", a.out, f"{Path(a.out).stat().st_size/1e6:.1f} MB")
    if a.fragment:
        Path(a.fragment).write_text(f"<title>Rearrangement Scenes from HSSD</title>{FONTS}<style>{CSS}</style><main>{body}</main>")
        print("wrote", a.fragment)


if __name__ == "__main__":
    main()
