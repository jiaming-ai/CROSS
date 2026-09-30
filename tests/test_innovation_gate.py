"""Chi-square innovation gate for proposals fused into hypothesis 0."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pp = pytest.importorskip("pypose")

from cross.core.config import HypothesisConfig, SystemConfig
from cross.core.hypothesis import HypothesisManager


def manager(gate):
    system = SimpleNamespace(device="cpu", topo_map=None, loaded_node_ids=frozenset(), config=SystemConfig())
    hm = HypothesisManager(system, 3, HypothesisConfig(h0_innovation_gate=gate, filter_process_std=0.01))
    hm.dist = (pp.identity_SE3(3), pp.se3(torch.ones(3, 6) * .05), torch.tensor([1., 0., 0.]))
    return hm


def proposal(x):
    pose = pp.identity_SE3()
    pose[0] = x
    return dict(pose=pose, std=pp.se3(torch.ones(6) * .05), score=torch.tensor(.8), source_indices=[(10, 0)])


@pytest.mark.parametrize("gate, x, fused", [(0., .8, True), (22.5, .8, False), (22.5, .05, True)])
def test_inconsistent_proposal_is_not_fused_into_hypothesis_0(gate, x, fused):
    hm = manager(gate)
    hm.align_proposal_prior([proposal(x)])
    audit = hm.last_alignment_audit[0]
    assert (audit["component"] == 0) == fused
