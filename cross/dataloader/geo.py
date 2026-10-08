"""GNSS fixes and compass samples of a prepared sequence folder, per frame (input of cross.geo through
obs["gnss"] / obs["compass"]).

Files (in the folder or its parent, the session folder of a prepared dataset):
- `gnss.txt`: `t lat_deg lon_deg alt_m [mode num_sats [hdop [sigma_h]]]`, t on the clock of `times.txt`.  A header
  that names the 5th column `msg` (NCLT: the NMEA sentence of the row, not a fix quality) makes the mode unknown.
- `mag.txt` (`t mag_x mag_y mag_z acc_x acc_y acc_z`) or a `ms25.txt` / `imu.txt` whose header names `mag_x` (NCLT layout:
  `t mag(3) acc(3) gyro(3)`), body frame forward-right-down; or `heading.txt` (`t yaw_enu_rad [sigma_rad]`).
- `times.txt`: frame times (seconds), one per frame; without it frames are idx / fps on the fix clock's start.

`window(prev_idx, idx, timestamp)` returns the latest fix in (time of frame prev_idx, time of frame idx] with its time
moved to the loader's clock (`timestamp` of the frame minus the fix's age), and the compass sample nearest the frame.
`degrade` (testing): extra white noise `sigma` (m), a constant `bias` (m, east), a fraction `drop` of fixes removed,
`outage` windows "a:b" (seconds from the start) without fixes."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np


def fill_altitude(t: np.ndarray, alt: np.ndarray, max_age: float = 1.0) -> np.ndarray:
    """Receivers that report the altitude in a separate sentence (NCLT: every other row): a fix without altitude
    takes the receiver's latest altitude of the last `max_age` seconds.  Horizontal-only factors let the optimisation
    pitch the trajectory (NCLT 2012-08-04: the end of the map rose by 21 m when half the factors had no altitude)."""
    out = alt.copy()
    last_t, last_a = -1e18, np.nan
    for i in range(len(t)):
        if np.isfinite(alt[i]):
            last_t, last_a = t[i], alt[i]
        elif t[i] - last_t <= max_age:
            out[i] = last_a
    return out


def _header(path: Path) -> str:
    with open(path) as f:
        lines = [ln for ln in (f.readline() for _ in range(5)) if ln.startswith("#")]
    return " ".join(lines).lower()


def _find(root: Path, name: str) -> Optional[Path]:
    for d in (root, root.parent):
        if (d / name).is_file():
            return d / name
    return None


class GeoStream:
    def __init__(self, fixes: np.ndarray, mode_known: bool, frame_times: np.ndarray, compass=None, degrade=None,
                 seed: int = 0):
        self.fix = fixes                       # (N, 8): t lat lon alt mode sats hdop sigma_h (nan when unknown)
        self.mode_known = mode_known
        self.frame_times = frame_times
        self.compass = compass                 # ("mag", t, mag, acc) or ("yaw", t, yaw, sigma) or None
        self.degrade = degrade or {}
        if self.degrade:
            self._apply_degrade(seed)

    @classmethod
    def load(cls, root, n_frames: int, fps: float, degrade: Optional[dict] = None, use_compass: bool = True,
             seed: int = 0) -> Optional["GeoStream"]:
        root = Path(root)
        gp = _find(root, "gnss.txt")
        if gp is None:
            return None
        h = _header(gp)
        g = np.loadtxt(gp, ndmin=2)
        fixes = np.full((len(g), 8), np.nan)
        fixes[:, :min(g.shape[1], 8)] = g[:, :8]
        mode_known = g.shape[1] > 4 and " msg" not in h
        if not mode_known:
            fixes[:, 4] = np.nan
        if " num_sats" in h and g.shape[1] > 5 and np.nanmax(g[:, 5]) <= 0:
            fixes[:, 5] = np.nan               # satellites not reported
        if not (g.shape[1] > 6 and "hdop" in h):
            fixes[:, 6] = np.nan
        if not (g.shape[1] > 7 and "sigma" in h):
            fixes[:, 7] = np.nan
        # NCLT writes two rows per fix (msg 2 without altitude, msg 3 with it): keep one per time stamp, prefer altitude
        order = np.lexsort((np.isnan(fixes[:, 3]), fixes[:, 0]))
        fixes = fixes[order]
        keep = np.r_[True, np.diff(fixes[:, 0]) > 1e-3]
        fixes = fixes[keep]
        fixes[:, 3] = fill_altitude(fixes[:, 0], fixes[:, 3])
        tp = root / "times.txt"
        times = np.loadtxt(tp).reshape(-1)[:n_frames] if tp.is_file() else fixes[0, 0] + np.arange(n_frames) / fps
        compass = None
        if use_compass:
            mp = _find(root, "mag.txt")
            ip = _find(root, "ms25.txt") or _find(root, "imu.txt")
            hp = _find(root, "heading.txt")
            if mp is not None:
                m = np.loadtxt(mp, ndmin=2)
                compass = ("mag", m[:, 0], m[:, 1:4], m[:, 4:7])
            elif ip is not None and "mag_x" in _header(ip):
                m = np.loadtxt(ip, ndmin=2)
                compass = ("mag", m[:, 0], m[:, 1:4], m[:, 4:7])
            elif hp is not None:
                m = np.loadtxt(hp, ndmin=2)
                compass = ("yaw", m[:, 0], m[:, 1], m[:, 2] if m.shape[1] > 2 else np.full(len(m), np.nan))
        gs = cls(fixes, mode_known, np.asarray(times, float), compass, degrade, seed)
        gs.loader_clock = np.arange(len(times)) / float(fps)     # the loaders' frame timestamps (idx / fps)
        return gs

    def _apply_degrade(self, seed: int):
        d = self.degrade
        rng = np.random.default_rng(seed)
        f = self.fix
        t0 = f[0, 0]
        keep = np.ones(len(f), bool)
        if d.get("drop"):
            keep &= rng.random(len(f)) >= float(d["drop"])
        for w in d.get("outage", []):
            a, b = w
            keep &= ~((f[:, 0] - t0 >= a) & (f[:, 0] - t0 < b))
        f = f[keep].copy()
        lat0 = math.radians(float(np.nanmedian(f[:, 1])))
        m_per_deg_lat = 111132.0
        m_per_deg_lon = 111320.0 * math.cos(lat0)
        e = np.zeros((len(f), 2))
        if d.get("sigma"):
            # a slowly wandering error (Gauss-Markov, 30 s), like a receiver's
            s = float(d["sigma"])
            a = math.exp(-1.0 / 30.0)
            for i in range(1, len(f)):
                dt = max(f[i, 0] - f[i - 1, 0], 0.0)
                ai = a ** dt
                e[i] = ai * e[i - 1] + math.sqrt(max(1 - ai * ai, 0.0)) * rng.normal(0, s, 2)
        if d.get("bias"):
            e[:, 0] += float(d["bias"])
        f[:, 1] += e[:, 1] / m_per_deg_lat
        f[:, 2] += e[:, 0] / m_per_deg_lon
        self.fix = f

    def window(self, prev_idx: Optional[int], idx: int, timestamp: float) -> dict:
        out = {}
        t1 = float(self.frame_times[idx])
        t0 = float(self.frame_times[prev_idx]) if prev_idx is not None and prev_idx >= 0 else t1 - 1.0
        lo = int(np.searchsorted(self.fix[:, 0], t0, side="right"))
        hi = int(np.searchsorted(self.fix[:, 0], t1, side="right"))
        if hi > lo:
            r = self.fix[hi - 1]
            # the fix's time on the loader's clock (piecewise linear between frames: dropped frames do not shift it)
            lc = getattr(self, "loader_clock", None)
            tf = float(np.interp(r[0], self.frame_times, lc)) + (float(timestamp) - float(lc[idx])) if lc is not None \
                else float(timestamp) - (t1 - float(r[0]))
            g = {"t": tf, "lat": float(r[1]), "lon": float(r[2]),
                 "alt": float(r[3]) if np.isfinite(r[3]) else None}
            if self.mode_known and np.isfinite(r[4]):
                g["mode"] = int(r[4])
            if np.isfinite(r[5]):
                g["num_sats"] = int(r[5])
            if np.isfinite(r[6]):
                g["hdop"] = float(r[6])
            if np.isfinite(r[7]):
                g["sigma_h"] = float(r[7])
            out["gnss"] = g
        if self.compass is not None:
            kind, t = self.compass[0], self.compass[1]
            j = int(np.clip(np.searchsorted(t, t1), 0, len(t) - 1))
            if abs(t[j] - t1) < 0.5:
                if kind == "mag":
                    out["compass"] = {"t": float(timestamp), "mag": self.compass[2][j].tolist(),
                                      "accel": self.compass[3][j].tolist()}
                else:
                    c = {"t": float(timestamp), "yaw": float(self.compass[2][j])}
                    if np.isfinite(self.compass[3][j]):
                        c["sigma_deg"] = math.degrees(float(self.compass[3][j]))
                    out["compass"] = c
        return out


def parse_degrade(spec: Optional[str]) -> dict:
    """'sigma=10,bias=15,drop=0.3,outage=60:180' -> dict (outage repeatable with ';' between windows)."""
    if not spec:
        return {}
    out = {}
    for kv in spec.split(","):
        k, v = kv.split("=", 1)
        if k == "outage":
            out["outage"] = [tuple(float(x) for x in w.split(":")) for w in v.split(";")]
        else:
            out[k] = float(v)
    return out


def attach(ds, root, args, query: bool) -> None:
    """Runner helper (--gnss, --gnss-degrade, --gnss-degrade-map, --no-compass): give the loader `ds` of the folder
    `root` its GNSS / compass stream (ds.geo), degraded as asked for this session."""
    if not getattr(args, "gnss", False) or getattr(args, "no_gnss", False):
        return
    n = len(ds)
    fps = float(getattr(ds, "fps", 10.0))
    spec = getattr(args, "gnss_degrade", None) if query else getattr(args, "gnss_degrade_map", None)
    seed = int(getattr(args, "seed", 0) or 0) + (1 if query else 0)
    ds.geo = GeoStream.load(root, n, fps, degrade=parse_degrade(spec), use_compass=not getattr(args, "no_compass", False),
                            seed=seed)
    if ds.geo is None:
        raise FileNotFoundError(f"--gnss: no gnss.txt in {root} or its parent")


def add_args(ap) -> None:
    ap.add_argument("--gnss", action="store_true", help="feed the folder's GNSS / compass data (gnss.txt, mag.txt / "
                    "ms25.txt with magnetometer, times.txt of the folder or its parent; see cross/dataloader/geo.py); geo "
                    "anchoring itself is on by default (geo.enabled)")
    ap.add_argument("--no-gnss", action="store_true", help="geo anchoring off (geo.enabled=false): no GNSS data, and a "
                    "loaded map's geo anchor is not kept on re-save")
    ap.add_argument("--gnss-degrade", default=None, help="degrade the query sessions' fixes: sigma=M,bias=M,drop=F,outage=A:B[;C:D]")
    ap.add_argument("--gnss-degrade-map", default=None, help="the same for the map session")
    ap.add_argument("--no-compass", action="store_true", help="with --gnss: ignore the magnetometer")
