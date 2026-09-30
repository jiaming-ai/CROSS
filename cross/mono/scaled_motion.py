"""Keep arbitrary-gauge motion separate from an available scale estimate."""

import numpy as np


class ScaledTranslation:
    """Integrate causal unit poses, holding translation until scale is available.

    The tracker gauge is anchored at its first camera. The first accepted
    scale therefore applies to the entire displacement accumulated so far.
    Later scale updates multiply new unit-gauge increments, as in the scalar
    frontend. Previously emitted poses are never modified. A relative-only
    experiment explicitly supplies scale=1 instead of claiming metric units.
    """

    def __init__(self):
        self.position = np.zeros(3)
        self.unit_position = np.zeros(3)
        self.scale_available = False

    def update(self, unit_position, scale=None):
        unit_position = np.asarray(unit_position, dtype=np.float64)
        if unit_position.shape != (3,) or not np.isfinite(unit_position).all():
            raise ValueError('Expected a finite three-coordinate unit position')
        if scale is not None and (not np.isfinite(scale) or scale <= 0):
            raise ValueError('Available scale must be finite and positive')
        if scale is None and self.scale_available:
            raise ValueError('An initialized scale cannot silently become unavailable')
        if scale is not None:
            if self.scale_available:
                self.position += scale * (unit_position - self.unit_position)
            else:
                self.position = scale * unit_position
            self.scale_available = True
        self.unit_position = unit_position.copy()
        return self.position.copy()
