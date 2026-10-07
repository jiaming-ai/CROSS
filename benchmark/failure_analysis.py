#!/usr/bin/env python3
"""Why systems fail: the trial analysis behind the Failure cases tab of the results page.

Joins every T3 trial outcome (benchmark/results/results.json, re-scored as the tables do) with the trial features of
benchmark/failure_stats.py (benchmark/results/failure_trials.json) and writes benchmark/results/failure_analysis.json:

  causes        failure rate of each system in trials with each candidate cause (rules in CAUSES), and the rate its
                matchability alone predicts (the system's failure rate per dataset and matchability bin, averaged over
                the cause's trials): a cause whose observed rate exceeds the prediction has an effect beyond lost
                local features
  matchability  failure rate per bin of inl_max (SIFT inliers against the map at the true place, best frame of the
                trial) and the odds ratio per 10x inliers of a logistic fit
  modes         how failed trials fail: no pose at all, or a pose more than 5x the threshold away
  overlap       trials whose last frame faces away from the map: failure rate by the best overlap within the trial
  localize      T2: time until the first lasting localization of a whole query session
  cameras       ROVER night: brightness and corners of the stereo and the colour camera
  orb_turns     ORB-SLAM3 mapping runs: tracking-loss onsets at turn rates above the session's 90th percentile

  python benchmark/failure_analysis.py
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

import build_site as bs
import make_tables as mt

ROOT = Path(__file__).resolve().parents[1]

# candidate causes: (id, group, label, rule text, predicate on a trial row)
CAUSES = [
    ("night", "appearance", "Night (ROVER)", "ROVER night and night-light sessions against the day map",
     lambda t: t["dataset"] == "rover" and t["cond"] in ("night", "night-light")),
    ("season", "appearance", "Other season (ROVER)", "ROVER summer, autumn, winter and spring sessions against the day map",
     lambda t: t["dataset"] == "rover" and t["cond"] in ("summer", "autumn", "winter", "spring")),
    ("dusk", "appearance", "Dusk, same season (ROVER)", "ROVER dusk session (the day before the map)",
     lambda t: t["dataset"] == "rover" and t["cond"] == "dusk"),
    ("lighting", "appearance", "Indoor lighting change", "SimChange light_* traversals, no other change",
     lambda t: t["dataset"] == "simchange" and t["cond"].startswith("light_") and "+" not in t["cond"]),
    ("objects", "appearance", "Moved or removed objects", "SimChange rearr / move / remove / background traversals, no other change",
     lambda t: t["dataset"] == "simchange" and any(x in t["cond"] for x in ("rearr", "move", "remove", "background")) and "+" not in t["cond"]),
    ("crowds", "appearance", "People and goods (market, cafe)", "OpenLORIS market and cafe",
     lambda t: t["dataset"] == "openloris" and t["scene"] in ("market", "cafe")),
    ("opposite", "viewpoint", "Facing away from the map", "the last frame's view differs by 90 deg or more from every map frame within the near radius",
     lambda t: t.get("end_view") is not None and t["end_view"] >= 90),
    ("offset", "viewpoint", "Sideways offset or turned camera", "SimChange offset / yaw / height traversals (not reversed)",
     lambda t: t["dataset"] == "simchange" and any(x in t["cond"] for x in ("offset", "yaw", "height")) and "reverse" not in t["cond"]),
    ("small_room", "viewpoint", "Small room, close range (office)", "OpenLORIS office (about 4 m across)",
     lambda t: t["dataset"] == "openloris" and t["scene"] == "office"),
    ("corridor", "ambiguity", "Straight corridor", "OpenLORIS corridor trials turning less than 30 deg in total",
     lambda t: t["scene"] == "corridor" and t.get("rot_deg", 1e9) < 30),
    ("standing", "ambiguity", "Standing still", "the robot moves in fewer than half of the trial's frames (mostly the start of a session)",
     lambda t: t.get("still", 0) > 0.5),
    ("fast_turn", "motion", "Fast turns", "peak turn rate over 1 s in the top 15 % of the dataset's trials",
     lambda t: t.get("fast_turn", False)),
]
INL_BINS = [(-1, 10, "< 10"), (10, 30, "10-30"), (30, 100, "30-100"), (100, 1e9, "≥ 100")]
STD_BINS = [-1, 10, 20, 30, 50, 100, 200, 1e9]


def condition(dataset, query):
    q = query.split("/")[-1]
    if dataset == "rover":
        return q.replace("campus_large_", "").split("_")[0]
    return q


def pct(xs):
    return None if not xs else round(100.0 * sum(xs) / len(xs), 1)


def logistic_or10(x, y):
    """Odds ratio of failure per 10x fewer inliers: logistic regression of failure on log(1 + inliers)."""
    X = np.c_[np.ones(len(x)), np.log1p(np.asarray(x, float))]
    y = np.asarray(y, float)
    w = np.zeros(2)
    for _ in range(60):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(2)
        w += np.linalg.solve(H, X.T @ (y - p))
    return round(math.exp(-w[1] * math.log(10)), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "benchmark/results/results.json"))
    ap.add_argument("--trials", default=str(ROOT / "benchmark/results/failure_trials.json"))
    ap.add_argument("--out", default=str(ROOT / "benchmark/results/failure_analysis.json"))
    a = ap.parse_args()
    ds_cfg, sy = mt.load()
    spec = yaml.safe_load((ROOT / "benchmark/configs/failure_cases.yaml").read_text())
    systems = spec["analysis_systems"]
    results = json.loads(Path(a.results).read_text())["results"]
    results = bs.prepare([r for r in results if not r.get("odom")], ds_cfg, sy, 0)
    stats = json.loads(Path(a.trials).read_text())
    feat = {(t["dataset"], t["query"], t["setup_dir"], int(t["start"])): t for t in stats["trials"]}
    # fast turns: top 15 % of the dataset's trials by peak turn rate
    p85 = {d: np.percentile([t["rate_max"] for t in stats["trials"] if t["dataset"] == d], 85)
           for d in {t["dataset"] for t in stats["trials"]}}
    rows = []                                    # one row per (system, trial)
    for r in results:
        k = f"{r['system']}|{r['setup']}"
        if r.get("track") != "t3" or r.get("status") != "ok" or k not in systems or not r.get("trials"):
            continue
        cfg = ds_cfg[r["dataset"]]
        su = cfg["setups"][r["setup"]]
        thr = cfg["thresholds"]
        for t in r["trials"]:
            if not t.get("counted", t.get("covered", True)):
                continue
            f = feat.get((r["dataset"], r["query"], su, int(t["start"])))
            if f is None:
                continue
            fe = t.get("final_err")
            rows.append({**f, "key": k, "scene": r["scene"], "cond": condition(r["dataset"], r["query"]),
                         "fail": not (fe is not None and fe < thr[1]), "err": fe, "thr": thr[1],
                         "fast_turn": f["rate_max"] > p85[r["dataset"]]})
    by_key = defaultdict(list)
    for x in rows:
        by_key[x["key"]].append(x)
    trial_id = lambda x: (x["dataset"], x["query"], x["start"])      # noqa: E731
    out = {"systems": [{"key": k, "label": mt.row_label(sy, *k.split("|"))} for k in systems],
           "n_trials": len({trial_id(x) for x in rows})}

    # ---- matchability curve and logistic odds ratio
    m = {"bins": [b[2] for b in INL_BINS], "n_trials": [], "fail": {}, "odds_per_10x": {}}
    for lo, hi, _ in INL_BINS:
        m["n_trials"].append(len({trial_id(x) for x in rows if lo < x["inl_max"] <= hi}))
    for k in systems:
        xs = by_key[k]
        m["fail"][k] = [pct([x["fail"] for x in xs if lo < x["inl_max"] <= hi]) for lo, hi, _ in INL_BINS]
        if len(xs) > 50:
            m["odds_per_10x"][k] = logistic_or10([x["inl_max"] for x in xs], [x["fail"] for x in xs])
    out["matchability"] = m

    # ---- causes: observed vs predicted from matchability
    def std_bin(v):
        return int(np.searchsorted(STD_BINS, v, side="left"))
    rate = {}
    for k in systems:
        acc = defaultdict(list)
        for x in by_key[k]:
            acc[(x["dataset"], std_bin(x["inl_max"]))].append(x["fail"])
        rate[k] = {kk: sum(v) / len(v) for kk, v in acc.items()}
    causes = []
    flagged = set()
    for cid, group, label, rule, pred in CAUSES:
        sel = [x for x in rows if pred(x)]
        ids = {trial_id(x) for x in sel}
        flagged |= ids
        c = {"id": cid, "group": group, "label": label, "rule": rule, "n_trials": len(ids),
             "inl_max_median": float(np.median([x["inl_max"] for x in sel])) if sel else None, "fail": {}, "expected": {}, "n": {}}
        for k in systems:
            xs = [x for x in sel if x["key"] == k]
            if len(xs) < 5:
                continue
            c["fail"][k] = pct([x["fail"] for x in xs])
            c["expected"][k] = round(100 * float(np.mean([rate[k][(x["dataset"], std_bin(x["inl_max"]))] for x in xs])), 1)
            c["n"][k] = len(xs)
        causes.append(c)
    for cid, label, sel in (("none", "None of these", [x for x in rows if trial_id(x) not in flagged]), ("all", "All trials", rows)):
        c = {"id": cid, "group": "reference", "label": label, "rule": "", "n_trials": len({trial_id(x) for x in sel}),
             "inl_max_median": float(np.median([x["inl_max"] for x in sel])), "fail": {}, "expected": {}, "n": {}}
        for k in systems:
            xs = [x for x in sel if x["key"] == k]
            if xs:
                c["fail"][k] = pct([x["fail"] for x in xs]); c["n"][k] = len(xs)
        causes.append(c)
    out["causes"] = causes

    # ---- failure modes
    modes = {}
    for k in systems:
        f = [x for x in by_key[k] if x["fail"]]
        if not f:
            continue
        errs = [x["err"] for x in f if x["err"] is not None]
        modes[k] = {"n_fail": len(f), "no_pose": pct([x["err"] is None for x in f]),
                    "far": pct([x["err"] is not None and x["err"] > 5 * x["thr"] for x in f]),
                    "err_median": round(float(np.median(errs)), 1) if errs else None}
    out["modes"] = modes

    # ---- overlap within the trial (last frame faces away from the map)
    ob = [(-1, 30, "a frame within 30 deg"), (30, 90, "best 30-90 deg"), (90, 181, "no frame within 90 deg")]
    ov = {"bins": [b[2] for b in ob], "n_trials": [], "fail": {}}
    away = [x for x in rows if x["end_view"] >= 90]
    for lo, hi, _ in ob:
        ov["n_trials"].append(len({trial_id(x) for x in away if lo < x["best_view"] <= hi}))
    for k in systems:
        ov["fail"][k] = [pct([x["fail"] for x in away if x["key"] == k and lo < x["best_view"] <= hi]) for lo, hi, _ in ob]
    out["overlap"] = ov

    # ---- T2: time to the first lasting localization
    loc = defaultdict(dict)
    for r in results:
        k = f"{r['system']}|{r['setup']}"
        if r.get("track") != "t2" or r.get("status") != "ok" or k not in systems:
            continue
        loc[k].setdefault(r["dataset"], []).append(r.get("time_to_localize"))
    out["localize"] = {k: {d: {"n": len(v), "median_s": (lambda t: None if not np.isfinite(t) else round(t / 10.0, 1))(
                               float(np.median([np.inf if x is None else x for x in v]))),
                               "within_10s": pct([x is not None and x <= 100 for x in v]), "never": pct([x is None for x in v])}
                           for d, v in per.items()} for k, per in loc.items()}

    # ---- ROVER night: the two cameras
    cams = {}
    for su in ("stereo", "rgbd"):
        xs = [t for t in stats["trials"] if t["dataset"] == "rover" and t["setup_dir"] == su
              and condition("rover", t["query"]) in ("night", "night-light") and t.get("bright") is not None]
        day = [t for t in stats["trials"] if t["dataset"] == "rover" and t["setup_dir"] == su
               and condition("rover", t["query"]) == "dusk" and t.get("bright") is not None]
        if xs:
            cams[su] = {"night_bright": round(float(np.median([t["bright"] for t in xs])), 1),
                        "night_corners": round(float(np.median([t["corners"] for t in xs])), 1),
                        "night_inl_max": round(float(np.median([t["inl_max"] for t in xs])), 1),
                        "dusk_bright": round(float(np.median([t["bright"] for t in day])), 1) if day else None,
                        "dusk_inl_max": round(float(np.median([t["inl_max"] for t in day])), 1) if day else None}
    out["cameras"] = cams

    # ---- ORB-SLAM3 mapping runs: tracking losses at fast turns
    turns = {}
    for r in stats.get("orb_tracking", []):
        if ds_cfg.get(r["dataset"], {}).get("dev"):
            continue
        t = turns.setdefault(r["setup"], {"runs": 0, "onsets": 0, "fast": 0, "rates": []})
        t["runs"] += 1
        for _, rate_ in r["onsets"]:
            t["onsets"] += 1; t["fast"] += int(rate_ > r["rate_p90"]); t["rates"].append(rate_)
    out["orb_turns"] = {su: {"runs": t["runs"], "onsets": t["onsets"], "fast_share": pct([1] * t["fast"] + [0] * (t["onsets"] - t["fast"])),
                             "median_rate": round(float(np.median(t["rates"])), 1) if t["rates"] else None} for su, t in turns.items()}

    # ---- CROSS: failures that end exactly where another CROSS variant ends (both still in their start frame)
    cr = {}
    errs = defaultdict(dict)
    for x in rows:
        if x["key"].startswith("cross_"):
            errs[trial_id(x)][x["key"]] = (x["err"], x["fail"])
    for k in [s for s in systems if s.startswith("cross_")]:
        f = [(tid, e[k][0]) for tid, e in errs.items() if k in e and e[k][1] and e[k][0] is not None]
        same = [tid for tid, err in f if any(kk != k and v[1] and v[0] is not None and abs(v[0] - err) < 0.3 for kk, v in errs[tid].items())]
        cr[k] = {"n_fail": len(f), "same_as_other_variant": len(same)}
    out["cross_unmerged_proxy"] = cr

    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {a.out}: {out['n_trials']} trials, {len(rows)} trial outcomes of {len(systems)} systems")
    for c in causes:
        print(f"  {c['label']:34s} n={c['n_trials']:4d} inl {c['inl_max_median']}: " +
              " ".join(f"{k.split('|')[0][:10]}={c['fail'].get(k)}/{c['expected'].get(k)}" for k in systems[:6]))


if __name__ == "__main__":
    main()
