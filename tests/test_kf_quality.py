"""Keyframe quality filter (cross/core/kf_quality.py): informative fraction of the view and its causes."""
import torch

from cross.core.config import KeyframeQualityConfig
from cross.core.kf_quality import KeyframeQuality


def _textured(seed, h=240, w=320):
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(3, h // 8, w // 8, generator=g)
    return torch.nn.functional.interpolate(base[None], size=(h, w), mode="bilinear", align_corners=False)[0]


def _filter(**kw):
    cfg = KeyframeQualityConfig(enabled=True, person=False, warmup=3, **kw)
    return KeyframeQuality(cfg, device="cpu")


def _seed(f, n=6, depth=None):
    for i in range(n):
        q = f.assess(_textured(i), depth)
        assert not q.junk


def test_textured_frames_are_kept():
    f = _filter()
    _seed(f)
    q = f.assess(_textured(100))
    assert not q.junk and q.info > 0.8


def test_blank_view_is_flat_junk():
    f = _filter()
    _seed(f)
    q = f.assess(torch.full((3, 240, 320), 0.5))
    assert q.junk and q.reason == "flat"


def test_dark_view_is_clipped_junk():
    f = _filter()
    _seed(f)
    q = f.assess(_textured(7) * 0.03)
    assert q.junk and q.reason == "clipped"


def test_half_occluded_is_not_junk_but_mostly_occluded_is():
    f = _filter()
    d = torch.full((1, 240, 320), 3.0)
    _seed(f, depth=d)
    near = d.clone()
    near[:, :, :160] = 0.4              # left half at 0.4 m
    q = f.assess(_textured(11), near)
    assert not q.junk and 0.4 < q.fractions["near"] < 0.6
    near[:, :, :280] = 0.4              # most of the view
    q = f.assess(_textured(12), near)
    assert q.junk and q.reason == "near"


def test_refine_with_pass_depth():
    f = _filter()
    _seed(f)
    q = f.assess(_textured(21))
    assert not q.junk
    d = torch.full((120, 160), 0.3)      # the pass sees an occluder over the whole view (another resolution)
    r = f.refine(q, d)
    assert r.junk and r.reason == "near" and r.stage == "pass"


def test_warmup_never_junk():
    f = _filter()
    q = f.assess(torch.full((3, 240, 320), 0.5))
    assert not q.junk


def test_stereo_near_field():
    import numpy as np
    f = _filter()
    f.set_stereo(np.array([[320.0, 0, 160], [0, 320.0, 120], [0, 0, 1]]), np.array([[1, 0, 0, 0.064], [0, 1, 0, 0],
                                                                                    [0, 0, 1, 0], [0, 0, 0, 1.0]]))
    g = torch.Generator().manual_seed(3)

    def pair(near_cols):
        left = torch.rand(1, 240, 320, generator=g).repeat(3, 1, 1)
        right = torch.roll(left, shifts=-4, dims=2)                       # background: 4 px (5.1 m)
        if near_cols:
            right[:, :, :near_cols] = torch.roll(left, shifts=-40, dims=2)[:, :, :near_cols]   # 40 px: 0.5 m
        return left, right

    for _ in range(6):                                                   # seed the session statistics
        f.assess(*pair(0)[:1], rgb_right=pair(0)[1])
    left, right = pair(0)
    q = f.assess(left, rgb_right=right)
    assert not q.junk and q.fractions["near"] < 0.1
    left, right = pair(300)
    q = f.assess(left, rgb_right=right)
    assert q.fractions["near"] > 0.6 and q.junk and q.reason == "near"


def test_add_person_keeps_threshold_and_skips_checked():
    f = _filter()
    _seed(f)
    q = f.assess(_textured(31), person=False)
    assert not q.person_checked
    r = f.add_person(q, _textured(31))         # the filter's config has person=False: unchanged
    assert r is q
