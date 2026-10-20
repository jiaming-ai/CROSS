import pytest

from cross_nav.context import past_frames, validate_lags

LAGS = [0, 4, 8, 16, 32, 64, 128, 256]


def test_exact_recipe_with_enough_history():
    anchor = (object(), object(), 300)
    frames = past_frames(anchor, LAGS)
    assert frames[0] == anchor
    assert [300 - f[2] for f in frames] == LAGS
    assert all(f[:2] == anchor[:2] for f in frames)


def test_early_history_is_unique_causal_and_fixed_count():
    for i in range(7, 257):
        frames = past_frames(("scene", "sequence", i), LAGS)
        indices = [f[2] for f in frames]
        assert len(set(indices)) == 8
        assert indices[0] == i
        assert all(0 <= k < i for k in indices[1:])


def test_insufficient_history_does_not_pad_with_future_or_duplicate_frames():
    assert past_frames(("s", "q", 6), LAGS) is None


@pytest.mark.parametrize("lags", [[], [1, 2], [0, 0], [0, -1], [0, 2, 1], [0, 1.5]])
def test_invalid_recipe_rejected(lags):
    with pytest.raises(ValueError):
        validate_lags(lags)


def test_causal_mode_rejects_bad_anchor_instead_of_swapping_to_past():
    from types import SimpleNamespace

    from cross_nav.data import NavStream

    frames = [(SimpleNamespace(world="world"), object(), i) for i in [10, 9]]
    pool = SimpleNamespace(name="test")
    for causal, expected in [(True, None), (False, 1)]:
        cfg = {"frames": [2, 2]}
        if causal:
            cfg["context_lags"] = [0, 1]
        stream = NavStream(cfg)
        stream.label_ds = {"test": "test"}
        stream.bevs = SimpleNamespace(get=lambda *args: {})
        stream._cam_h = lambda wb, sc, sq, i: float("nan") if i == 10 else 1.0
        result = stream._frame0(None, pool, frames)
        assert (result[3] if result else None) == expected


def test_context_count_must_match_batch_shape():
    from cross_nav.data import NavStream

    with pytest.raises(ValueError, match="fixed matching"):
        NavStream({"frames": [6, 10], "context_lags": LAGS})
