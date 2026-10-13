"""Causal temporal proxies for recent and older keyframes (no pose-based selection)."""
from __future__ import annotations


def validate_lags(lags):
    lags = tuple(lags)
    if not lags or lags[0] != 0 or any(int(x) != x or x < 0 for x in lags):
        raise ValueError("context_lags must start at zero and contain nonnegative integers")
    if any(a >= b for a, b in zip(lags, lags[1:])):
        raise ValueError("context_lags must be strictly increasing")
    return tuple(map(int, lags))


def past_frames(anchor, lags):
    """Keep anchor fixed; fill clipped early-history gaps without future frames or repeats.

    Return None if there are too few distinct past frames for the requested count.
    The world-level target is built separately and may use a wider time range.
    """
    lags = validate_lags(lags)
    scene, sequence, index = anchor
    if index < len(lags) - 1:
        return None
    selected = {min(index, lag) for lag in lags}
    while len(selected) < len(lags):
        ordered = sorted(selected)
        left, right = max(zip(ordered, ordered[1:]), key=lambda p: (p[1] - p[0], -p[0]))
        assert right - left > 1
        selected.add((left + right) // 2)
    return [(scene, sequence, index - lag) for lag in sorted(selected)]
