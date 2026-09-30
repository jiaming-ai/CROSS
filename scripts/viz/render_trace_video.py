#!/usr/bin/env python3
"""Render an MP4 of the map construction / relocalization replay from a recorded trace.

Same content as the interactive page (build_trace_page.py): observation image and retrieved
keyframes on the left, the keyframe graph aligned to ground truth in the centre (keyframes, edges,
hypotheses, ground-truth and estimated trajectories of every session in its own colour) and the
position-error timeline at the bottom.  Frames are rendered with matplotlib in parallel chunks and
concatenated with ffmpeg.

Usage:
  python scripts/viz/render_trace_video.py --trace outputs/viz/lonemonk/trace --out outputs/viz/lonemonk/replay.mp4 \
      --stride 2 --fps 20 --workers 16 [--sessions 0 1 2]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_trace_page import build, pose7_to_T  # noqa: E402

# colours of the page (light theme)
C = dict(bg="#eceef1", panel="#f9fafb", line="#d3d8e0", line2="#e2e6ec", ink="#161b22", ink2="#4f5866", ink3="#8a93a1",
         accent="#c8451f", odom="#9aa3b2", visual="#7f9cc2", lc="#d33b3b", temp="#b8bfc9")
SES_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ERRMAX = 8.0


def xf(p, T):
    M = T @ pose7_to_T(p)
    return M[0, 3], M[1, 3], M[2, 3], M[0, 2], M[1, 2]   # x, y, z, forward (camera +z) projected


class Renderer:
    def __init__(self, D: dict, frames_root: Path, align: int, metric: int = 2, width=1920, height=1080):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.D, self.dir = D, frames_root
        self.S, self.SES, self.N, self.E = D["steps"], D["sessions"], D["nodes"], D["edges"]
        self.K = D["n_components"]
        self.T = np.asarray(D["T_first"] if align == 0 else D["T_umeyama"])
        self.align = align
        self.metric = metric
        self.metric_name = ["vs GT (first-keyframe alignment)", "vs GT (Umeyama alignment)", "map-relative (CROSS protocol; mapping: drift vs GT)"][metric]
        self.gt = np.array([xf(st["gt"], np.eye(4)) for st in self.S])
        self.est = np.array([xf(st["mu"][0], self.T) for st in self.S])
        self.node_ids = sorted(self.N.keys())
        x0, y0 = self.gt[:, :2].min(0) - 3; x1, y1 = self.gt[:, :2].max(0) + 3
        self.xlim, self.ylim = (x0, x1), (y0, y1)
        self.fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=C["bg"])
        from matplotlib import font_manager
        fams = {f.name for f in font_manager.fontManager.ttflist}
        plt.rcParams["font.family"] = ["IBM Plex Sans"] if "IBM Plex Sans" in fams else ["DejaVu Sans"]
        self.glyph = 0.012 * max(self.xlim[1] - self.xlim[0], self.ylim[1] - self.ylim[0])
        # layout: left rail 0.21, map, bottom timeline 0.14
        self.ax_img = self.fig.add_axes([0.012, 0.655, 0.196, 0.265])
        self.ax_ret = [self.fig.add_axes([0.012 + (i % 3) * 0.066, 0.505 - (i // 3) * 0.10, 0.062, 0.09]) for i in range(6)]
        self.ax_hyp = self.fig.add_axes([0.034, 0.20, 0.07, 0.12])
        self.ax_map = self.fig.add_axes([0.225, 0.17, 0.765, 0.765])
        self.ax_tl = self.fig.add_axes([0.05, 0.035, 0.94, 0.10])
        self._img_cache = {}

    # ------------------------------------------------------------------ helpers
    def img_for(self, variant, f):
        p = self.dir / "frames" / self.D["scene"] / variant / f"{f:06d}.jpg"
        if not p.is_file():
            return None
        if p not in self._img_cache:
            from PIL import Image
            if len(self._img_cache) > 64:
                self._img_cache.clear()
            self._img_cache[p] = np.asarray(Image.open(p).convert("RGB"))
        return self._img_cache[p]

    def img(self, t):
        return self.img_for(self.ses_of(t)["variant"], self.S[t]["f"])

    def ses_of(self, t):
        for s in self.SES:
            if s["start_step"] <= t < s["end_step"]:
                return s
        return self.SES[-1]

    def node_entry(self, nd, t):
        e = nd["hist"][0]
        for h in nd["hist"]:
            if h[0] <= t:
                e = h
            else:
                break
        return e

    def edge_visible(self, e, t):
        if e.get("lc") is not None and e["lc"] <= t:
            return True
        return e["st"] <= t < e["en"]

    # ------------------------------------------------------------------ frame
    def draw(self, t):
        plt = self.plt
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Polygon
        S, SES, N, E = self.S, self.SES, self.N, self.E
        st, ses = S[t], self.ses_of(t)
        col = SES_COL[ses["id"] % 8]
        for ax in [self.ax_img, self.ax_hyp, self.ax_map, self.ax_tl, *self.ax_ret]:
            ax.clear(); ax.set_facecolor(C["bg"])
        fig = self.fig
        for txt in list(fig.texts):
            txt.remove()

        # ---- header
        from build_trace_page import METHOD_LABELS
        fig.text(0.012, 0.975, f"{METHOD_LABELS.get(self.D['method'], self.D['method'])} · {self.D['scene']}", fontsize=14, weight="semibold", color=C["ink"], va="center")
        fig.text(0.012, 0.952, f"map {self.D['map_variant']}" + (f" · stereo baseline {self.D['baseline']:.2f} m" if self.D.get('baseline') else "") + (f" · odometry SNR {self.D['snr']:g}" if self.D.get('snr') else "") +
                 f" · alignment: {'first keyframe' if self.align == 0 else 'Umeyama'}", fontsize=9.5, color=C["ink2"], va="center")
        x = 0.40
        for s in SES:
            cur = s["id"] == ses["id"]
            name = ("map · " if s["kind"] == "map" else "") + (s["variant"] if len(s["variant"]) <= 16 else s["variant"][:15] + "…")
            fig.text(x, 0.965, "●", color=SES_COL[s["id"] % 8], fontsize=11, va="center", alpha=1 if s["start_step"] <= t else .35)
            fig.text(x + 0.009, 0.965, name, fontsize=9.5, va="center", color=C["ink"] if cur else C["ink2"], weight="semibold" if cur else "normal")
            x += 0.016 + 0.0058 * len(name)

        # ---- observation
        im = self.img(t)
        if im is not None:
            self.ax_img.imshow(im)
        self.ax_img.set_xticks([]); self.ax_img.set_yticks([])
        for sp in self.ax_img.spines.values():
            sp.set_visible(False)
        self.ax_img.set_title("OBSERVATION (LEFT CAMERA)", loc="left", fontsize=8.5, color=C["ink3"], pad=4)
        fig.text(0.012, 0.638, f"{ses['variant']} · frame {st['f']}", fontsize=9.5, color=C["ink2"], va="center", family="monospace")
        fig.text(0.208, 0.638, "not localized" if st.get("lost") else ("observed" if st["obs"] else "odometry only"), fontsize=9.5, ha="right", va="center",
                 color=C["lc"] if st.get("lost") else (C["ink"] if st["obs"] else C["ink3"]), weight="semibold" if st["obs"] else "normal")
        # ---- retrieved keyframes
        fig.text(0.012, 0.612, "RETRIEVED KEYFRAMES", fontsize=8.5, color=C["ink3"], va="center")
        ret = sorted(st.get("ret", []), key=lambda r: -r[1])[:6] if not self.D.get("external") else []
        used = {v[0]: v[1] for v in st.get("vk", [])}
        for i, ax in enumerate(self.ax_ret):
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i < len(ret):
                kid, sc = ret[i]
                nd = N.get(kid)
                thumb = self.img_for(SES[nd["s"]]["variant"], nd["f"]) if nd else None
                if thumb is not None:
                    ax.imshow(thumb, alpha=1 if kid in used else .4, cmap=None)
                    if kid not in used:
                        ax.imshow(thumb.mean(2), cmap="gray", alpha=.6)
                lab = f"KF {kid % 100000}  {sc:.2f}" + ((f"  inl {used[kid]:.0f}" if used[kid] >= 10 else f"  cov {used[kid]:.2f}") if kid in used else "  ✕")
                ax.text(0.03, 0.06, lab, transform=ax.transAxes, fontsize=6.5, color="white",
                        bbox=dict(facecolor="black", alpha=.55, pad=1.5, edgecolor="none"))
            else:
                ax.set_facecolor(C["panel"])
        if not ret:
            fig.text(0.012, 0.585, "not recorded for external systems" if self.D.get("external") else ("no observation at this step (odometry propagates the belief)" if not st["obs"] else "nothing retrieved"), fontsize=8.5, color=C["ink3"], va="center")
        # ---- hypotheses
        ax = self.ax_hyp
        G = self.gt[t]
        fig.text(0.012, 0.335, "POSE HYPOTHESES", fontsize=8.5, color=C["ink3"], va="center")
        for k in range(self.K):
            w = st["w"][k]; dead = w < 1e-3
            p = xf(st["mu"][k], self.T)
            err = math.sqrt((p[0] - G[0]) ** 2 + (p[1] - G[1]) ** 2 + (p[2] - G[2]) ** 2)
            y = self.K - 1 - k
            ax.barh(y, 1, height=.5, color=C["line2"])
            ax.barh(y, w, height=.5, color=col if k == 0 else C["ink3"])
            ax.text(-0.06, y, f"h{k}", ha="right", va="center", fontsize=8.5, color=C["ink3"] if dead else C["ink"], family="monospace", clip_on=False)
            ax.text(1.06, y, "free" if dead else f"w {w:.2f}  err {err:.2f} m  {'realized' if st['rl'][k] else 'tracking'}",
                    va="center", fontsize=8, color=C["ink3"] if dead else C["ink"], family="monospace", clip_on=False)
        ax.set_xlim(0, 1); ax.set_ylim(-.6, self.K - .4); ax.axis("off")

        # ---- map
        ax = self.ax_map
        ax.set_facecolor(C["bg"])
        ax.set_xlim(*self.xlim); ax.set_ylim(*self.ylim); ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, color=C["line"], lw=.6)
        ax.tick_params(colors=C["ink3"], labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(C["line"])
        ax.set_xlabel("x [m]", color=C["ink3"], fontsize=8); ax.set_ylabel("y [m]", color=C["ink3"], fontsize=8)
        s = ses
        to = min(t, s["end_step"] - 1) + 1; c = SES_COL[s["id"] % 8]
        ax.plot(self.gt[s["start_step"]:to, 0], self.gt[s["start_step"]:to, 1], ls=(0, (4, 3)), lw=1.6, color=c, alpha=.85, zorder=2)
        frm = s["merge_step"] if (s.get("merge_step") is not None and t >= s["merge_step"]) else s["start_step"]
        segs_e = []
        cur_seg = []
        for i in range(frm, to):
            if S[i].get("lost") or (cur_seg and np.hypot(self.est[i, 0] - self.est[i - 1, 0], self.est[i, 1] - self.est[i - 1, 1]) > 2.0):
                if len(cur_seg) > 1:
                    segs_e.append(cur_seg)
                cur_seg = [] if S[i].get("lost") else [(self.est[i, 0], self.est[i, 1])]
                continue
            cur_seg.append((self.est[i, 0], self.est[i, 1]))
        if len(cur_seg) > 1:
            segs_e.append(cur_seg)
        for sg in segs_e:
            sg = np.asarray(sg)
            ax.plot(sg[:, 0], sg[:, 1], lw=2.2, color=c, alpha=.9, zorder=3)
        pos = {}
        n_mis = 0
        for nid in self.node_ids:
            nd = N[nid]
            if nd["step"] <= t < nd["end"]:
                en = self.node_entry(nd, t)
                if len(en) > 2 and en[2]:
                    n_mis += 1
                    continue
                pos[nid] = xf(en[1], self.T)
        segs = {"odom": [], "vis0": [], "vish": [], "lc": []}
        for e in E:
            if not self.edge_visible(e, t):
                continue
            a, b = pos.get(e["a"]), pos.get(e["b"])
            if a is None or b is None:
                continue
            key = "lc" if (e.get("lc") is not None and e["lc"] <= t) else ("odom" if e["t"] == "odom" else ("vis0" if e["c"] == 0 else "vish"))
            segs[key].append([(a[0], a[1]), (b[0], b[1])])
        ax.add_collection(LineCollection(segs["vish"], colors=C["visual"], lw=.6, alpha=.25, ls=(0, (3, 3)), zorder=4))
        ax.add_collection(LineCollection(segs["vis0"], colors=C["visual"], lw=.7, alpha=.4, zorder=4))
        ax.add_collection(LineCollection(segs["odom"], colors=C["odom"], lw=1.1, alpha=.9, zorder=5))
        ax.add_collection(LineCollection(segs["lc"], colors=C["lc"], lw=1.8, zorder=6))
        if st.get("vk") and not st.get("lost"):
            p0 = self.est[t]
            for kid, conf, _ in st["vk"]:
                q = pos.get(kid)
                if q:
                    ax.plot([p0[0], q[0]], [p0[1], q[1]], color=col, lw=.8, ls=(0, (2, 3)), alpha=.35 + .5 * min(1.0, conf / (100.0 if conf > 1 else 1.0)), zorder=7)
                    ax.scatter([q[0]], [q[1]], s=70, facecolors="none", edgecolors=col, lw=1.2, zorder=9)
        perm = [(p[0], p[1], nid) for nid, p in pos.items() if N[nid]["perm"]]
        tmp = [(p[0], p[1]) for nid, p in pos.items() if not N[nid]["perm"]]
        if tmp:
            ax.scatter(*zip(*tmp), s=9, facecolors="none", edgecolors=C["temp"], lw=.9, zorder=8)
        if perm:
            xs, ys, ids = zip(*perm)
            ax.scatter(xs, ys, s=18, c=[C["ink"] if N[i]["s"] == 0 else SES_COL[N[i]["s"] % 8] for i in ids], edgecolors=C["bg"], lw=.6, zorder=9)
        if st.get("nk") is not None and st["nk"] in pos:
            q = pos[st["nk"]]
            ax.scatter([q[0]], [q[1]], s=140, facecolors="none", edgecolors=C["ink"], lw=1.6, zorder=10)
        for k in range(self.K - 1, -1, -1):
            w = st["w"][k]
            if w < 1e-3 or st.get("lost"):
                continue
            p = xf(st["mu"][k], self.T)
            if k == 0:
                self._camera(ax, p, self.glyph * (0.7 + 0.5 * math.sqrt(w)), col, zorder=12)
            else:
                ax.scatter([p[0]], [p[1]], s=60 + 500 * w, facecolors=C["ink2"], alpha=.18, edgecolors=C["ink2"], lw=1.6 if st["rl"][k] else 1, ls="-" if st["rl"][k] else "--", zorder=11)
                ax.text(p[0] + .4, p[1] + .3, f"h{k}", fontsize=7.5, color=C["ink2"], family="monospace", zorder=11)
        self._camera(ax, self.gt[t], self.glyph * 1.1, C["accent"], zorder=13)
        # status box
        n_perm = sum(1 for i in pos if N[i]["perm"]); n_tmp = len(pos) - n_perm
        n_lc = sum(1 for i in range(ses["start_step"], t + 1) if "pgo" in S[i])
        er = st["err"]; o = self.metric * 4
        ext = self.D.get("external")
        errs = "–" if er[o] is None else f"{er[o]:.2f} m · {er[o + 1]:.1f}°"
        lines = [f"{'session':<14}{'mapping' if ses['kind'] == 'map' else 'relocalization':>16}", f"{'variant':<14}{ses['variant'][:16]:>16}",
                 f"{'frame':<14}{str(st['f']) + ' / ' + str(ses['n_frames']):>16}", f"{'step':<14}{('not localized' if st.get('lost') else ('observation' if st['obs'] else 'odometry only')):>16}",
                 f"{'keyframes':<14}{(str(n_perm) + ' map samples') if ext else (str(n_perm) + ' + ' + str(n_tmp) + ' temp'):>16}", f"{'edges':<14}{sum(len(v) for v in segs.values()):>16}"]
        if not ext:
            lines += [f"{'loop closures':<14}{n_lc:>16}", f"{'misplaced hid.':<14}{n_mis:>16}"]
        lines.append(f"{'c0 error':<14}{errs:>16}")
        if not ext:
            lines += [f"{'best (h' + str(st['kb']) + ')':<14}{f'{er[o + 2]:.2f} m · {er[o + 3]:.1f}°':>16}", f"{'weight w0':<14}{st['w'][0]:>16.2f}"]
        ax.text(0.985, 0.98, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=8.2, family="monospace", color=C["ink"],
                bbox=dict(facecolor=C["panel"], edgecolor=C["line"], alpha=.92, boxstyle="round,pad=0.5"), zorder=20)
        if "pgo" in st or st.get("lc") is not None:
            msg = f"Loop closure: hypothesis {st['pgo']['hypo']} merged, {st['pgo']['n_opt']} keyframes optimised" if "pgo" in st else f"Loop closure detected (hypothesis {st['lc']})"
            ax.text(0.5, 0.03, msg, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="white", weight="semibold",
                    bbox=dict(facecolor=C["lc"], edgecolor="none", boxstyle="round,pad=0.45"), zorder=21)
        # legend
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=C["ink2"], ls=(0, (4, 3)), lw=1.6, label="ground truth"), Line2D([], [], color=C["ink2"], lw=2.2, label="belief (component 0)"),
                   Line2D([], [], marker="o", color="none", markerfacecolor=C["ink"], markersize=5, label="permanent keyframe"),
                   Line2D([], [], marker="o", color="none", markerfacecolor="none", markeredgecolor=C["temp"], markersize=4, label="temporary keyframe"),
                   Line2D([], [], color=C["odom"], lw=1.1, label="odometry edge"), Line2D([], [], color=C["visual"], lw=.8, label="visual edge"),
                   Line2D([], [], color=C["lc"], lw=1.8, label="loop-closure edge"),
                   Line2D([], [], marker="o", color="none", markerfacecolor="none", markeredgecolor=C["ink2"], markersize=7, label="other hypothesis (size ∝ weight)"),
                   Line2D([], [], marker=">", color="none", markerfacecolor=C["accent"], markersize=7, label="ground-truth camera")]
        ax.legend(handles=handles, loc="lower left", fontsize=7.5, frameon=True, facecolor=C["panel"], edgecolor=C["line"], labelcolor=C["ink2"])

        # ---- timeline
        ax = self.ax_tl
        NT = len(S)
        for s in SES:
            ax.axvspan(s["start_step"], s["end_step"], color=SES_COL[s["id"] % 8], alpha=.10, lw=0)
            ax.text(s["start_step"] + 4, ERRMAX - .3, s["variant"], fontsize=7.5, color=C["ink3"], va="top", clip_on=True)
            e = np.array([min(S[i]["err"][o], ERRMAX) if S[i]["err"][o] is not None else np.nan for i in range(s["start_step"], s["end_step"])])
            ax.plot(np.arange(s["start_step"], s["end_step"]), e, color=SES_COL[s["id"] % 8], lw=1.2)
        kf_steps = [N[i]["step"] for i in self.node_ids if N[i]["perm"]] if not self.D.get("external") else []
        ax.vlines(kf_steps, -0.05, 0.3, color=C["ink3"], lw=.5, alpha=.7)
        pg = [i for i in range(NT) if "pgo" in S[i]]
        ax.scatter(pg, [ERRMAX + .6] * len(pg), marker="D", s=18, color=C["lc"], clip_on=False, zorder=5)
        ax.axvline(t, color=C["ink"], lw=1.2)
        ax.set_xlim(0, NT - 1); ax.set_ylim(-0.3, ERRMAX + .2)
        ax.set_yticks([0, 2, 4, 6, 8]); ax.set_yticklabels(["0", "2", "4", "6", "8 m"], fontsize=7.5, color=C["ink3"])
        ax.set_xticks([]); ax.grid(True, axis="y", color=C["line"], lw=.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"position error of the belief (component 0), {self.metric_name} · ticks: permanent keyframes · diamonds: loop closure + PGO", loc="left", fontsize=8, color=C["ink2"], pad=3)
        fig.text(0.99, 0.139, f"step {t + 1} / {NT}", fontsize=8.5, color=C["ink2"], ha="right", va="bottom", family="monospace")

    def _camera(self, ax, p, size, color, zorder):
        from matplotlib.patches import Polygon
        a = math.atan2(p[4], p[3])
        pts = np.array([[1, 0], [-.8, .65], [-.35, 0], [-.8, -.65]]) * size
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        pts = pts @ R.T + np.array([p[0], p[1]])
        ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=C["panel"], lw=1, zorder=zorder))

    def render_chunk(self, frames, out_path, fps):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{int(self.fig.get_figwidth() * 100)}x{int(self.fig.get_figheight() * 100)}",
               "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", str(out_path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for t in frames:
            self.draw(t)
            self.fig.canvas.draw()
            buf = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
            proc.stdin.write(np.ascontiguousarray(buf).tobytes())
        proc.stdin.close(); proc.wait()


def _worker(args):
    D, frames_root, align, metric, frames, out_path, fps = args
    r = Renderer(D, Path(frames_root), align, metric)
    r.render_chunk(frames, out_path, fps)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="trace directory (trace.json)")
    ap.add_argument("--frames-root", default="outputs/viz/page", help="page directory holding frames/<scene>/<variant>/*.jpg (build_trace_page.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=2, help="trace steps per video frame")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--align", type=int, default=0, help="map->GT alignment for the drawing: 0 first keyframe, 1 Umeyama")
    ap.add_argument("--metric", type=int, default=2, help="error metric of the timeline: 0 vs GT first-kf, 1 vs GT Umeyama, 2 map-relative")
    ap.add_argument("--sessions", type=int, nargs="*", default=None, help="restrict to these session ids")
    ap.add_argument("--hold", type=int, default=15, help="extra frames held at loop closures and session ends")
    args = ap.parse_args()

    trace_dir = Path(args.trace)
    trace = json.loads((trace_dir / "trace.json").read_text())
    trace.setdefault("method", trace_dir.name[len("trace_"):] if trace_dir.name.startswith("trace_") else "cross_stereo")
    D = build(trace)
    D["nodes"] = {int(k): v for k, v in D["nodes"].items()}
    S, SES = D["steps"], D["sessions"]
    frames = []
    for s in SES:
        if args.sessions is not None and s["id"] not in args.sessions:
            continue
        for t in range(s["start_step"], s["end_step"], args.stride):
            frames.append(t)
            if any("pgo" in S[u] for u in range(t, min(t + args.stride, s["end_step"]))):
                frames.extend([t] * args.hold)
        frames.extend([s["end_step"] - 1] * args.hold)
    n = len(frames)
    nw = max(1, min(args.workers, n // 40 + 1))
    chunks = [frames[i * n // nw:(i + 1) * n // nw] for i in range(nw)]
    tmp = Path(tempfile.mkdtemp(prefix="trace_video_"))
    jobs = [(D, args.frames_root, args.align, args.metric, ch, str(tmp / f"chunk{i:03d}.mp4"), args.fps) for i, ch in enumerate(chunks) if ch]
    print(f"{n} frames in {len(jobs)} chunks -> {args.out}")
    import multiprocessing as mp
    with mp.get_context("fork").Pool(len(jobs)) as pool:
        parts = pool.map(_worker, jobs)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", args.out], check=True)
    for p in parts:
        os.remove(p)
    os.remove(lst); tmp.rmdir()
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 2**20:.1f} MB, {n / args.fps:.0f} s)")


if __name__ == "__main__":
    main()
