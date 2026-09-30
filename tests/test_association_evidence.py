"""Loaded-map association evidence for sparse (monocular) verification."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pp = pytest.importorskip("pypose")

from cross.core.config import HypothesisConfig, SystemConfig
from cross.core.hypothesis import HypothesisManager


def manager(**kw):
    system = SimpleNamespace(device="cpu", topo_map=None, loaded_node_ids=frozenset(), config=SystemConfig())
    cfg = HypothesisConfig(chart_aware=True, session_recovery=True, reloc_association_evidence=True, **kw)
    hm = HypothesisManager(system, 2, cfg)
    hm.dist = (pp.identity_SE3(2), pp.se3(torch.ones(2, 6) * .1), torch.tensor([.2, .8]))
    hm.start_tracking_chart()
    hm.component_charts[1] = 7          # map chart, hypothesis 0 is the new session's chart
    hm.reference_support.start({1, 2, 3})
    return hm


def test_association_components_only_while_unjoined():
    hm = manager()
    assert hm.association_components().tolist() == [False, True]
    hm.reference_support.mark_anchored()
    assert hm.association_components().tolist() == [False, False]


def test_sparse_consistent_map_detections_commit_but_one_reference_does_not():
    hm = manager()
    hm.realized[:] = True
    hm.hist_valid[1] = True
    # three verified frames (two distinct map keyframes) among five missed detections
    hits = {0: {1}, 3: {2}, 6: {1}}
    for step in range(8):
        hm.reference_support.observe(step, {k: (0, 1) for k in hits.get(step, ())})
        hm.log_c_hist[1, step] = 2.5 if step in hits else -0.36
    assert hm.reference_audit(1)["eligible"]
    assert hm.detect_loop_closure({})["loop_closure"]
    hm2 = manager()
    hm2.realized[:] = True
    hm2.hist_valid[1] = True
    for step in range(8):
        hm2.reference_support.observe(step, {1: (0, 1)} if step in hits else {})
        hm2.log_c_hist[1, step] = 2.5 if step in hits else -0.36
    assert not hm2.reference_audit(1)["eligible"]      # a single map keyframe is not redundant support
    assert not hm2.detect_loop_closure({})["loop_closure"]


def test_inconsistent_detections_do_not_commit():
    hm = manager()
    hm.realized[:] = True
    hm.hist_valid[1] = True
    for step in range(8):
        hm.reference_support.observe(step, {1 + step % 3: (0, 1)})
        hm.log_c_hist[1, step] = -4.0      # verified edges that contradict the branch's motion
    assert not hm.detect_loop_closure({})["loop_closure"]
