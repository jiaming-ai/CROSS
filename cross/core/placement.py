"""Placement hypotheses of a relocalization session (mapping.hypothesis.reloc_placement_places).

Before a session is joined to the stored map, its placement in the map is (nearly) one rigid transform A: the
session's odometry is rigid over the session.  Every raw map measurement (the relative pose T_ref_cur of the current
frame to a stored keyframe) implies one:  A = T_map_ref @ T_ref_cur @ inv(T_session_cur).  Measurements of the true
place agree on one A, which they confirm from many separate places along the walk; look-alike places scatter, each
confirmed from the one spot where it looks alike (iPhone capture 1: the true placement of a relocalization session is
supported from 4-11 places 10 m apart, look-alike corridors from 1-2).

The tracker keeps every placement hypothesis that measurements have proposed (allowing slow drift: a track follows its
measurements), counts the separate places that support each, and calls a placement confirmed once it is supported from
`places` places and no other placement is.  The best track is the session's provisional placement from the first
measurement on; merges into the stored map wait for a confirmed placement and must agree with it.
"""
from __future__ import annotations

import json
import math
import os
from typing import List, Optional

import numpy as np


_REC = []


def _recorder():
    """Debug: CROSS_PLACEMENT_RECORD=<dir> writes every tracker input (odometry, map measurements, merge candidates) to
    <dir>/placement_<pid>.jsonl for offline replay of decision rules (one file per process, shared by its sessions)."""
    d = os.environ.get("CROSS_PLACEMENT_RECORD")
    if not d:
        return None
    if not _REC:
        import atexit, sys
        os.makedirs(d, exist_ok=True)
        f = open(os.path.join(d, f"placement_{os.getpid()}.jsonl"), "a", buffering=1 << 16)
        f.write(json.dumps({"e": "argv", "argv": sys.argv}) + "\n")
        atexit.register(f.close)
        _REC.append(f)
    return _REC[0]


def _m(T) -> list:
    return [round(float(x), 5) for x in np.asarray(T, float)[:3, :].reshape(-1)]


class _Track:
    __slots__ = ("A", "yaw", "places", "steps", "n_meas", "last_step", "last_pos")

    def __init__(self, A, yaw, place, step):
        self.A, self.yaw = A, yaw
        self.places = [place]
        self.steps = {step}
        self.n_meas = 1
        self.last_step = step
        self.last_pos = place


class PlacementTracker:
    """drift: the tolerances grow by this fraction of the distance walked (session odometry) since a track's last
    measurement (metres per metre, and the same number in degrees per metre), so that one placement stays one track
    over a long walk in a map that is slightly bent or scaled against the odometry.  dominance: a placement is confirmed
    when it has at least `places` places and `dominance` times the places of every different placement."""

    def __init__(self, places: int, sep_m: float, yaw_tol_deg: float, t_tol_m: float, up, drift: float = 0.1,
                 dominance: float = 2.0, blend: float = 0.5):
        self.min_places, self.sep = int(places), float(sep_m)
        self.yaw_tol, self.t_tol, self.blend = math.radians(yaw_tol_deg), float(t_tol_m), float(blend)
        self.drift, self.dominance = float(drift), float(dominance)
        u = np.asarray(up, float).reshape(3)
        u = u / max(np.linalg.norm(u), 1e-9)
        x = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        e1 = x - (x @ u) * u
        self.u, self.e1 = u, e1 / np.linalg.norm(e1)
        self.e2 = np.cross(self.u, self.e1)
        self.rec = _recorder()
        self.record(e="init", up=[float(x) for x in self.u])
        self.reset()

    def reset(self):
        self.T = np.eye(4)                    # session odometry (camera to session start)
        self.tracks: List[_Track] = []
        self.confirmed_step: Optional[int] = None
        self.current: Optional[_Track] = None    # the placement the session joined the map with (commit())
        self._conf_track: Optional[_Track] = None  # confirmed() (analysis): kept until a different one overtakes it
        if getattr(self, "rec", None) is not None:
            self.rec.write('{"e": "reset"}\n')

    # ------------------------------------------------------------------ inputs
    def odometry(self, delta, step: Optional[int] = None):
        """The step's odometry increment (4x4, T_current = T_previous @ delta)."""
        if delta is not None:
            self.T = self.T @ np.asarray(delta, float)
        if self.rec is not None:
            self.rec.write(json.dumps({"e": "odo", "s": step, "T": _m(self.T)}) + "\n")

    def record(self, **ev):
        if self.rec is not None:
            self.rec.write(json.dumps(ev) + "\n")

    def _yaw(self, R) -> float:
        v = R @ self.e1
        return math.atan2(float(v @ self.e2), float(v @ self.e1))

    def _h(self, t) -> np.ndarray:
        return np.array([t @ self.e1, t @ self.e2])

    def implied(self, T_map_cur) -> np.ndarray:
        """The session placement that a map-frame pose of the current frame implies."""
        return np.asarray(T_map_cur, float) @ np.linalg.inv(self.T)

    def _dist(self, y, h, tr: _Track, d_walk: float, k: float = 1.0) -> Optional[float]:
        """Normalised distance of a placement (yaw y, horizontal translation h) to a track, None beyond k tolerances."""
        ty = k * (self.yaw_tol + math.radians(self.drift) * d_walk)
        tt = k * (self.t_tol + self.drift * d_walk)
        dy = abs((y - tr.yaw + math.pi) % (2 * math.pi) - math.pi)
        dt = float(np.linalg.norm(h - self._h(tr.A[:3, 3])))
        if dy >= ty or dt >= tt:
            return None
        return dt / tt + dy / ty

    def _matches(self, A, pos):
        y, h = self._yaw(A[:3, :3]), self._h(A[:3, 3])
        out = []
        for tr in self.tracks:
            d = self._dist(y, h, tr, float(np.linalg.norm(pos - tr.last_pos)))
            if d is not None:
                out.append((d, tr))
        return sorted(out, key=lambda x: x[0])

    def find(self, A) -> Optional[_Track]:
        m = self._matches(A, self.T[:3, 3])
        return m[0][1] if m else None

    def add(self, T_map_ref, T_ref_cur, step: int, kf: Optional[int] = None):
        """A raw map measurement of the current frame."""
        if self.rec is not None:
            self.rec.write(json.dumps({"e": "meas", "s": step, "kf": kf, "Tmr": _m(T_map_ref), "Trc": _m(T_ref_cur),
                                       "T": _m(self.T)}) + "\n")
        A = np.asarray(T_map_ref, float) @ np.asarray(T_ref_cur, float) @ np.linalg.inv(self.T)
        place = self.T[:3, 3].copy()
        m = self._matches(A, place)
        if not m:
            self.tracks.append(_Track(A, self._yaw(A[:3, :3]), place, step))
            return
        tr = m[0][1]
        for _, other in m[1:]:                 # one measurement joins tracks: they were pieces of one placement
            for p in other.places:
                if all(np.linalg.norm(p - q) > self.sep for q in tr.places):
                    tr.places.append(p)
            tr.steps |= other.steps
            tr.n_meas += other.n_meas
            if other is self.current:
                self.current = tr
            if other is self._conf_track:
                self._conf_track = tr
            self.tracks.remove(other)
        # follow drift: the track's placement moves to its latest measurements
        tr.A = tr.A.copy()
        tr.A[:3, 3] = (1 - self.blend) * tr.A[:3, 3] + self.blend * A[:3, 3]
        dy = (self._yaw(A[:3, :3]) - tr.yaw + math.pi) % (2 * math.pi) - math.pi
        tr.yaw = tr.yaw + self.blend * dy
        if all(np.linalg.norm(place - p) > self.sep for p in tr.places):
            tr.places.append(place)
        tr.steps.add(step)
        tr.n_meas += 1
        tr.last_step = step
        tr.last_pos = place

    # ------------------------------------------------------------------ decisions
    def _apart(self, a: _Track, b: _Track) -> bool:
        """Different placements, not pieces of one drifting placement: more than 3 (drift-grown) tolerances apart."""
        return self._dist(b.yaw, self._h(b.A[:3, 3]), a, float(np.linalg.norm(a.last_pos - b.last_pos)), k=3.0) is None

    def ranked(self) -> List[_Track]:
        return sorted(self.tracks, key=lambda t: (-len(t.places), -len(t.steps)))

    def provisional(self) -> Optional[_Track]:
        r = self.ranked()
        return r[0] if r else None

    def confirmed(self, step: Optional[int] = None) -> Optional[_Track]:
        r = self.ranked()
        if self._conf_track is not None:
            # hysteresis: the confirmed placement stays until a different placement has more places
            if not any(len(o.places) > len(self._conf_track.places) and self._apart(self._conf_track, o) for o in r
                       if o is not self._conf_track):
                return self._conf_track
            self._conf_track = None
        if not r or len(r[0].places) < self.min_places:
            return None
        rival = max((len(o.places) for o in r[1:] if self._apart(r[0], o)), default=0)
        if len(r[0].places) < self.dominance * rival:
            return None                           # another placement is nearly as well supported: ambiguous
        if self.confirmed_step is None and step is not None:
            self.confirmed_step = int(step)
        self._conf_track = r[0]
        return r[0]

    def decide(self, A, min_steps: int = 3, min_places: int = 1):
        """Merge decision for a candidate whose implied session placement is A: (allowed, reason, track).
        - The session already joined the map (`current`): the candidate must agree with that placement.
        - Otherwise: merge at once when the candidate's placement has at least min_steps supporting steps and
          `dominance` times the steps of every different placement; wait while placements compete (look-alike places)."""
        tr = self.find(A)
        if tr is None:
            return False, "no support", None
        if self.current is not None:
            if tr is self.current or not self._apart(self.current, tr):
                return True, "agrees with the joined placement", tr
            return False, "disagrees with the joined placement", tr
        if len(tr.steps) < min_steps or len(tr.places) < min_places:
            return False, "too little support", tr
        rivals = [o for o in self.tracks if o is not tr and self._apart(tr, o)]
        r_steps = max((len(o.steps) for o in rivals), default=0)
        r_places = max((len(o.places) for o in rivals), default=0)
        if len(tr.steps) < self.dominance * r_steps or (r_places and len(tr.places) < self.dominance * r_places):
            return False, "competing placements", tr
        return True, "dominant", tr

    def commit(self, tr):
        """The session joined the map with placement track tr."""
        if tr is not None:
            self.current = tr

    def summary(self, k: int = 4) -> list:
        return [{"yaw_deg": round(math.degrees(t.yaw), 1), "t": [round(float(x), 1) for x in t.A[:3, 3]],
                 "places": len(t.places), "steps": len(t.steps)} for t in self.ranked()[:k]]
