"""16-bit PNG depth codec shared by the converters and the loader.

Log encoding: value v in 1..65535 <-> depth d = exp(log(dmin) + (v - 1) / 65534 * log(dmax / dmin)), 0 = invalid.
With dmin = 0.01 and dmax = 1000 (scene units) one step is a relative change of 1.8e-4, so the same encoding serves
indoor, outdoor and object-scale scenes; depths outside [dmin, dmax] (e.g. sky) are stored as invalid.
"""
from __future__ import annotations

import numpy as np

DMIN, DMAX = 0.01, 1000.0
_LR = np.log(DMAX / DMIN)


def encode(depth: np.ndarray, dmin: float = DMIN, dmax: float = DMAX) -> np.ndarray:
    d = np.asarray(depth, np.float64)
    ok = np.isfinite(d) & (d >= dmin * (1 - 1e-6)) & (d <= dmax)
    v = np.zeros(d.shape, np.uint16)
    v[ok] = np.clip(np.round((np.log(np.maximum(d[ok], dmin)) - np.log(dmin)) / np.log(dmax / dmin) * 65534) + 1, 1, 65535)
    return v


def decode(v: np.ndarray, dmin: float = DMIN, dmax: float = DMAX) -> np.ndarray:
    v = np.asarray(v)
    d = np.exp(np.log(dmin) + (v.astype(np.float32) - 1) / 65534.0 * np.float32(np.log(dmax / dmin)))
    d[v == 0] = 0
    return d.astype(np.float32)
