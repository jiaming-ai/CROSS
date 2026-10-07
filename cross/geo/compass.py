"""Compass (magnetometer) heading with disturbance detection and online offset calibration.

Heading convention: ENU yaw (radians, from East, counter-clockwise) of the body's forward axis.  A compass bearing
(clockwise from north) b converts to yaw = pi/2 - b.

A magnetometer measures the local field, which steel structures, motors and electronics distort (indoors, near
vehicles).  A sample is disturbed when the field magnitude or the inclination (dip) departs from the reference by
more than the chi-square level times the robust spread; the reference is learned online from samples taken while the
GNSS gate is passing fixes (outdoors), or from all samples until then.  The heading offset between the compass and the
camera's forward axis in ENU (magnetic declination + mounting yaw) is calibrated online against the GNSS-anchored map
(robust circular mean); before that the compass gives no absolute heading unless `offset_deg` is configured.  Heading
changes are also checked against the odometry's yaw changes (drift free over short spans)."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import chi2


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def tilt_compensated_yaw(mag, accel, frame: str = "frd") -> float:
    """ENU yaw of the body's forward (x) axis from a magnetometer and an accelerometer (specific force, at rest ~ -g
    in the body frame for 'frd' = forward-right-down, +g up component for 'flu' = forward-left-up), relative to
    magnetic north (declination not applied)."""
    m = np.asarray(mag, float)
    a = np.asarray(accel, float)
    if frame == "flu":                           # convert to forward-right-down
        m = m * np.array([1, -1, -1])
        a = a * np.array([1, -1, -1])
    # in FRD at rest the accelerometer measures -g along +down -> a = (0, 0, -9.81); gravity direction g_b = -a
    g = -a / max(np.linalg.norm(a), 1e-9)
    roll = math.atan2(g[1], g[2])
    pitch = math.atan2(-g[0], math.hypot(g[1], g[2]))
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    # level the magnetic vector (rotate body -> local NED-aligned horizontal frame)
    mx = m[0] * cp + m[1] * sr * sp + m[2] * cr * sp
    my = m[1] * cr - m[2] * sr
    bearing = math.atan2(-my, mx)                # clockwise from magnetic north
    return float(wrap(math.pi / 2 - bearing))


def levelled_field(mag, accel, frame: str = "frd") -> np.ndarray:
    """Horizontal components (north-ish x along the body's forward projection, y to its right) of the magnetic field,
    levelled with roll / pitch from the accelerometer (forward-right-down body frame)."""
    m = np.asarray(mag, float)
    a = np.asarray(accel, float)
    if frame == "flu":
        m = m * np.array([1, -1, -1])
        a = a * np.array([1, -1, -1])
    g = -a / max(np.linalg.norm(a), 1e-9)
    roll = math.atan2(g[1], g[2])
    pitch = math.atan2(-g[0], math.hypot(g[1], g[2]))
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    return np.array([m[0] * cp + m[1] * sr * sp + m[2] * cr * sp, m[1] * cr - m[2] * sr])


def fit_circle(xy: np.ndarray):
    """Algebraic (Kasa) circle fit: (centre (2,), radius)."""
    x, y = xy[:, 0], xy[:, 1]
    A = np.c_[x, y, np.ones(len(x))]
    b = -(x ** 2 + y ** 2)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    c = np.array([-D / 2, -E / 2])
    return c, math.sqrt(max(c @ c - F, 0.0))


def field_inclination(mag, accel, frame: str = "frd"):
    """(|B|, inclination) of the field: inclination = angle between B and the horizontal plane (positive down)."""
    m = np.asarray(mag, float)
    a = np.asarray(accel, float)
    if frame == "flu":
        m = m * np.array([1, -1, -1])
        a = a * np.array([1, -1, -1])
    g = -a / max(np.linalg.norm(a), 1e-9)       # down direction in the body frame
    nb = float(np.linalg.norm(m))
    return nb, float(math.asin(np.clip(np.dot(m, g) / max(nb, 1e-12), -1, 1)))


@dataclass
class CompassConfig:
    confidence: float = 0.999           # chi-square level (None in GeoConfig: the verified loop closure's)
    sigma_deg: float = 5.0              # prior heading noise of an undisturbed compass
    offset_deg: Optional[float] = None  # known compass->camera heading offset (declination + mounting); None: calibrate
    min_offset_samples: int = 30        # samples needed before the calibrated offset is used
    reference_window: int = 2000        # samples kept for the field reference
    frame: str = "frd"                  # body frame of raw magnetometer / accelerometer samples
    hard_iron: bool = False             # online hard-iron calibration (NCLT 27 sessions: no gain, p50 4.1 vs 4.0 deg)


class Compass:
    def __init__(self, cfg: CompassConfig = None):
        self.cfg = cfg or CompassConfig()
        self.k1 = math.sqrt(chi2.ppf(self.cfg.confidence, 1))
        self.ref_good = deque(maxlen=self.cfg.reference_window)    # (|B|, dip) while GNSS is good
        self.ref_all = deque(maxlen=self.cfg.reference_window)
        self.offset_samples = deque(maxlen=2000)                    # (compass yaw - camera yaw in ENU)
        self.offset = None if self.cfg.offset_deg is None else math.radians(self.cfg.offset_deg)
        self.offset_std = None if self.offset is None else 0.0
        self.stats = {"samples": 0, "disturbed": 0, "used": 0}
        self.ref_fixed = None                                       # (median, spread) stored with a map
        # hard-iron calibration: levelled horizontal field samples of good conditions binned by heading; the circle
        # through them is centred on the hard-iron offset once the robot has turned through most headings
        self.hi_bins = {}
        self.hard_iron = None

    def _reference(self):
        if len(self.ref_good) < 50 and self.ref_fixed is not None:
            return self.ref_fixed
        ref = self.ref_good if len(self.ref_good) >= 50 else self.ref_all
        if len(ref) < 20:
            return None
        a = np.asarray(ref)
        med = np.median(a, 0)
        mad = 1.4826 * np.median(np.abs(a - med), 0)
        return med, np.maximum(mad, [1e-3 * max(med[0], 1e-6), math.radians(0.5)])

    def disturbed(self, mag, accel, outdoor_ok: bool = False) -> bool:
        """True when the field departs from the reference (magnitude or inclination).  `outdoor_ok`: the GNSS gate is
        passing fixes, so the sample also feeds the reference of good conditions."""
        nb, dip = field_inclination(mag, accel, self.cfg.frame)
        ref = self._reference()
        bad = False
        if ref is not None:
            med, mad = ref
            bad = abs(nb - med[0]) > self.k1 * mad[0] or abs(dip - med[1]) > self.k1 * mad[1]
        self.ref_all.append((nb, dip))
        if outdoor_ok:
            # every sample of good conditions feeds the reference (its median / MAD are robust to the few disturbed
            # ones; keeping only accepted samples would let the band shrink onto itself)
            self.ref_good.append((nb, dip))
        return bad

    def heading(self, sample: dict, outdoor_ok: bool = False):
        """(yaw_enu_of_compass_body, sigma) or None.  `sample`: {"yaw": rad} / {"bearing_deg": deg} (heading given by
        the sensor; its own sigma optional), or {"mag": (3,), "accel": (3,)} (raw)."""
        self.stats["samples"] += 1
        sigma = math.radians(sample.get("sigma_deg", self.cfg.sigma_deg))
        if "mag" in sample and sample.get("mag") is not None:
            if self.disturbed(sample["mag"], sample["accel"], outdoor_ok):
                self.stats["disturbed"] += 1
                return None
            h = levelled_field(sample["mag"], sample["accel"], self.cfg.frame)
            if outdoor_ok and self.cfg.hard_iron:
                self._add_hard_iron_sample(h)
            if self.hard_iron is not None:
                h = h - self.hard_iron
            yaw = float(wrap(math.pi / 2 - math.atan2(-h[1], h[0])))
        elif "yaw" in sample:
            yaw = float(sample["yaw"])
        elif "bearing_deg" in sample:
            yaw = float(wrap(math.pi / 2 - math.radians(sample["bearing_deg"])))
        else:
            return None
        self.stats["used"] += 1
        return yaw, sigma

    def _add_hard_iron_sample(self, h: np.ndarray, n_bins: int = 36, per_bin: int = 20):
        k = int(((math.atan2(h[1], h[0]) + math.pi) / (2 * math.pi)) * n_bins) % n_bins
        b = self.hi_bins.setdefault(k, deque(maxlen=per_bin))
        b.append(h.copy())
        if len(self.hi_bins) >= int(0.75 * n_bins) and self.stats["samples"] % 50 == 0:
            xy = np.concatenate([np.asarray(v) for v in self.hi_bins.values()])
            c, r = fit_circle(xy)
            if r > 0 and np.linalg.norm(c) < r:          # a plausible offset (smaller than the field itself)
                self.hard_iron = c

    def add_offset_sample(self, compass_yaw: float, camera_yaw_enu: float):
        """A heading pair while the map is anchored and GNSS is good: calibrates the compass offset."""
        self.offset_samples.append(float(wrap(compass_yaw - camera_yaw_enu)))
        if self.cfg.offset_deg is not None or len(self.offset_samples) < self.cfg.min_offset_samples:
            return
        a = np.asarray(self.offset_samples)
        # robust circular mean: median around the circular mean, then MAD
        c = math.atan2(np.sin(a).mean(), np.cos(a).mean())
        d = wrap(a - c)
        med = float(np.median(d))
        mad = 1.4826 * float(np.median(np.abs(d - med)))
        keep = np.abs(d - med) <= self.k1 * max(mad, math.radians(1.0))
        self.offset = float(wrap(c + d[keep].mean()))
        self.offset_std = float(d[keep].std() / math.sqrt(max(keep.sum(), 1))) if keep.sum() > 1 else None
        self.offset_spread = float(d[keep].std()) if keep.sum() > 1 else None

    def camera_yaw(self, compass_yaw: float) -> Optional[float]:
        """ENU yaw of the camera's forward axis from a compass heading, once the offset is known."""
        if self.offset is None:
            return None
        return float(wrap(compass_yaw - self.offset))

    def state(self) -> dict:
        return {"offset": self.offset, "offset_std": self.offset_std,
                "hard_iron": None if self.hard_iron is None else [float(x) for x in self.hard_iron],
                "offset_spread": getattr(self, "offset_spread", None), "stats": dict(self.stats),
                "reference": None if self._reference() is None else [list(map(float, x)) for x in self._reference()]}

    def load_state(self, s: dict):
        if s.get("offset") is not None and self.cfg.offset_deg is None:
            self.offset = float(s["offset"])
            self.offset_std = s.get("offset_std")
            self.offset_spread = s.get("offset_spread")
        if s.get("hard_iron") is not None:
            self.hard_iron = np.asarray(s["hard_iron"], float)
        ref = s.get("reference")
        if ref:
            self.ref_fixed = (np.asarray(ref[0], float), np.asarray(ref[1], float))
