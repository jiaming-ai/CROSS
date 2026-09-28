"""CPU checks for retrieval provenance and failed geometric proposals."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pp = pytest.importorskip("pypose")


def test_alignment_audit_distinguishes_loaded_sources_without_changing_assignment():
    from cross.core.hypothesis import HypothesisManager

    system = SimpleNamespace(device="cpu", loaded_node_ids=frozenset({4}))
    manager = HypothesisManager(system, 3)
    manager.dist = (pp.identity_SE3(3), pp.se3(torch.ones(3, 6) * .1), torch.tensor([1., 0., 0.]))
    remote = pp.identity_SE3()
    remote[0] = 5.
    proposals = [dict(pose=pp.identity_SE3(), std=pp.se3(torch.ones(6) * .1),
                      score=torch.tensor(.8), source_indices=[(9, 0)]),
                 dict(pose=remote, std=pp.se3(torch.ones(6) * .1),
                      score=torch.tensor(.6), source_indices=[(4, 0)])]
    _, _, _, _, edges = manager.align_proposal_prior(proposals)
    assert edges == {9: (0, 0), 4: (0, 1)}
    local, historical = manager.last_alignment_audit
    assert local["component"] == 0 and local["action"] == "matched"
    assert not local["sources"][0]["loaded"]
    assert historical["component"] == 1 and historical["action"] == "born"
    assert historical["sources"] == [dict(keyframe_id=4, source_component=0, loaded=True)]
    assert historical["nearest_prior_distance"] == pytest.approx(5.)


def test_pair_audit_keeps_rejections_in_input_order():
    from cross.mono.retrieval import MetricRelativePose

    estimator = MetricRelativePose.__new__(MetricRelativePose)
    estimator.device = "cpu"
    estimator.features = lambda rgb: rgb
    identity = np.eye(4)
    bad_inverse = identity.copy()
    bad_inverse[0, 3] = 1.
    # Three candidates: forward failure, cycle failure, accepted.
    estimates = iter([None, (identity, 30, .2), (bad_inverse, 25, .3),
                      (identity, 80, .2), (identity, 70, .3)])
    def estimate(*_):
        value = next(estimates)
        estimator.refiner.last_match_audit = dict(reason="accepted" if value else "few_matches")
        return value
    estimator.refiner = SimpleNamespace(estimate=estimate)
    images = torch.zeros(3, 3, 8, 8)
    depth = torch.ones(3, 8, 8)
    poses, valid, confidence = estimator.estimate_pose(images, depth, images[0], depth[0])
    assert valid.tolist() == [False, False, True]
    assert poses.shape == (1, 7) and confidence.shape == (1,)
    assert [row["reason"] for row in estimator.last_pair_audit] == [
        "forward_pnp", "inconsistent_cycle", "accepted"]
    assert estimator.last_pair_audit[0]["forward"]["reason"] == "few_matches"
    assert estimator.last_pair_audit[1]["cycle_translation_m"] == 1.
    # No stale rejection history when the next frame has one accepted pair.
    estimates = iter([(identity, 40, .1), (identity, 40, .1)])
    estimator.estimate_pose(images[:1], depth[:1], images[0], depth[0])
    assert len(estimator.last_pair_audit) == 1 and estimator.last_pair_audit[0]["accepted"]


@pytest.mark.parametrize("distance", [.1, 2., 10.])
def test_historical_recovery_keeps_temporal_gates_independent_of_chart_distance(distance):
    from cross.core.config import HypothesisConfig
    from cross.core.hypothesis import HypothesisManager

    manager = HypothesisManager(SimpleNamespace(device="cpu"), 2, HypothesisConfig(session_recovery=True))
    poses = pp.identity_SE3(2)
    poses[1, 0] = distance
    manager.dist = (poses, pp.se3(torch.ones(2, 6) * .1), torch.tensor([.5, .5]))
    manager.realized[:] = True
    manager.log_c_hist[1] = 2.
    manager.reference_support.start({1, 2})
    for i in range(4):
        manager.reference_support.observe(i, {1: (0, 1), 2: (0, 1)})
    assert manager.detect_loop_closure({})["loop_closure"]
    manager.log_c_hist[1] = -2.
    assert not manager.detect_loop_closure({})["loop_closure"]  # history still mandatory
    manager.log_c_hist[1] = 2.
    manager.reference_support.observe(4, {1: (0, 1)}, newborns=[1])
    assert not manager.detect_loop_closure({})["loop_closure"]  # even when distance >3m
    manager.reference_support.mark_anchored()
    assert manager.detect_loop_closure({})["loop_closure"] == (distance >= 3.)


def test_historical_slot_preserves_budget_thresholds_scores_and_embedding_cost():
    from cross.db.db import KeyframeDatabase

    database = KeyframeDatabase.__new__(KeyframeDatabase)
    database._current_size, database.top_k = 5, 3
    database._embedding_buffer = torch.tensor([[.95], [.9], [.8], [.6], [.1]])
    database.score_threshold_high = database.score_threshold_low = .3
    database._keyframe_by_atlas = {None: [SimpleNamespace(id=i) for i in range(5)]}
    database._index_to_atlas_idx = {i: (None, i) for i in range(5)}
    calls = []
    def embedding(image):
        calls.append(image)
        return torch.ones(1)
    database.vpr_model = SimpleNamespace(get_embedding=embedding)
    original = database.query(None)
    balanced = database.query(None, reserved_keyframe_ids={3, 4}, reserved_count=1)
    low_score_only = database.query(None, reserved_keyframe_ids={4}, reserved_count=1)
    assert [k.id for k in original["keyframes"]] == [0, 1, 2]
    assert [k.id for k in balanced["keyframes"]] == [0, 1, 3]
    assert balanced["scores"] == pytest.approx([.95, .9, .6])
    assert [k.id for k in low_score_only["keyframes"]] == [0, 1, 2]
    assert len(calls) == 3
