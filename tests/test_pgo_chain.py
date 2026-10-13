"""chain_poses (cross/core/pgo.py): the memoised initialisation of untwinned session keyframes in a merge graph gives
bit-identical poses to composing each keyframe's walk separately (the previous code, kept here as the reference)."""
import random

import pypose as pp
import torch

from cross.core.pgo import chain_poses


class _E:
    def __init__(self, mean):
        self.mean = mean


class _V:
    def __init__(self, pose):
        self.pose = pose


def _reference(start_ids, twins, successor, odom_edges, limit=10000):
    out = {}
    for s in start_ids:
        chain, cur, ok = [], s, False
        for _ in range(limit):
            nxt = successor.get(cur)
            if nxt is None:
                break
            chain.append((cur, nxt))
            cur = nxt
            if cur in twins:
                ok = True
                break
        if ok:
            T = twins[cur].pose
            for (a, b) in reversed(chain):
                T = T @ odom_edges[(a, b)].mean.Inv()
            out[s] = T
    return out


def _graph(seed, n=300, p_twin=0.05, gaps=3):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    ids = sorted(rng.sample(range(3 * n), n))
    odom = {}
    for a, b in zip(ids, ids[1:]):
        odom[(a, b)] = _E(pp.randn_SE3(sigma=0.3))
    for _ in range(gaps):                         # breaks in the chain (no odometry edge)
        k = rng.randrange(len(ids) - 1)
        odom.pop((ids[k], ids[k + 1]), None)
    for _ in range(5):                            # extra edges out of a keyframe (only the first is followed)
        k = rng.randrange(len(ids) - 3)
        odom[(ids[k], ids[k + 2])] = _E(pp.randn_SE3(sigma=0.3))
    successor = {}
    for (a, b) in odom:
        successor.setdefault(a, b)
    twins = {i: _V(pp.randn_SE3()) for i in ids if rng.random() < p_twin}
    starts = [i for i in ids if i not in twins]
    rng.shuffle(starts)
    return starts, twins, successor, odom


def _same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k].tensor(), b[k].tensor()), k


def test_chain_poses_bit_identical():
    for seed in range(6):
        starts, twins, successor, odom = _graph(seed)
        _same(chain_poses(starts, twins, successor, odom), _reference(starts, twins, successor, odom))
        # sorted starts (the order of the graph's vertices) as well
        s2 = sorted(starts)
        _same(chain_poses(s2, twins, successor, odom), _reference(s2, twins, successor, odom))


def test_chain_poses_limit_and_no_twin():
    starts, twins, successor, odom = _graph(7, n=200, p_twin=0.02, gaps=0)
    for limit in (1, 5, 40):
        _same(chain_poses(starts, twins, successor, odom, limit=limit),
              _reference(starts, twins, successor, odom, limit=limit))
    _same(chain_poses(starts, {}, successor, odom), {})


def test_chain_poses_cycle():
    odom = {(1, 2): _E(pp.randn_SE3()), (2, 3): _E(pp.randn_SE3()), (3, 1): _E(pp.randn_SE3())}
    successor = {1: 2, 2: 3, 3: 1}
    assert chain_poses([1, 2, 3], {}, successor, odom) == {}
