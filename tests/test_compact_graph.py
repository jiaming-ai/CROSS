"""Compact keyframe / edge fields (cross.core.types.pack_tensor): same values and semantics as the tensors they hold."""
import copy
import pickle

import numpy as np
import pypose as pp
import pytest
import torch

from cross.core.types import Edge, EdgeType, Keyframe, VisualEdge, pack_tensor, unpack_tensor


def _kf():
    return Keyframe(pose_mu=pp.randn_SE3(5), pose_std=pp.se3(torch.rand(5, 6)), pose_weights=torch.rand(5),
                    timestamp=1.0, temporary=True)


def test_pack_roundtrip_values_and_types():
    for v in (pp.randn_SE3(), pp.randn_SE3(5), pp.se3(torch.rand(6)), torch.rand(5), torch.arange(5)):
        arr, lt = pack_tensor(v)
        assert isinstance(arr, np.ndarray)
        out = unpack_tensor(arr, lt)
        assert type(out) is type(v) and out.dtype == v.dtype and out.shape == v.shape
        assert torch.equal(out.tensor() if isinstance(out, pp.LieTensor) else out,
                           v.tensor() if isinstance(v, pp.LieTensor) else v)
        if isinstance(v, pp.LieTensor):
            assert out.ltype is v.ltype
    assert pack_tensor(None) == (None, None)
    p = torch.nn.Parameter(torch.rand(3))
    assert pack_tensor(p)[0] is p                     # kept as it is
    g = torch.rand(3, requires_grad=True)
    assert pack_tensor(g)[0] is g


def test_keyframe_fields_compact_and_write_through():
    kf = _kf()
    assert isinstance(kf.__dict__["_pose_mu"], np.ndarray)
    before = kf.pose_mu.clone()
    mu = kf.pose_mu
    assert isinstance(mu, pp.LieTensor) and mu.ltype is pp.SE3_type and mu.shape == (5, 7)
    kf.pose_mu[1] = pp.identity_SE3()                 # in place, as the hypothesis manager does
    assert torch.equal(kf.pose_mu[1].tensor(), pp.identity_SE3().tensor())
    assert torch.equal(kf.pose_mu[0].tensor(), before[0].tensor())
    with torch._C.DisableTorchFunctionSubclass():     # the PGO write-back path
        kf.pose_mu[0] = torch.zeros(7)
    assert float(kf.pose_mu[0].tensor().abs().sum()) == 0.0
    kf.pose_weights[2] = 0.0
    assert float(kf.pose_weights[2]) == 0.0
    old = kf.pose_std
    kf.pose_std = pp.se3(torch.zeros(5, 6))           # reassignment does not touch a tensor read before
    assert float(old.tensor().abs().sum()) > 0
    assert kf.pose_charts is None
    kf.pose_charts = torch.zeros(5, dtype=torch.int64)
    assert kf.pose_charts.dtype == torch.int64


def test_keyframe_pickle_and_ids():
    kf = _kf()
    kf2 = pickle.loads(pickle.dumps(kf))
    assert kf2.id == kf.id and torch.equal(kf2.pose_mu.tensor(), kf.pose_mu.tensor())


def test_edge_slots_meta_and_np_cache():
    e = Edge(pp.randn_SE3(), pp.se3(torch.rand(6)), EdgeType.ODOMETRY)
    assert not hasattr(e, "odom_fault") and getattr(e, "noise_scale", None) is None
    e.n_frames = 3
    e.odom_fault = 0.2
    e.anything = "x"                                  # unknown metadata still works (__dict__)
    m = e.mean_np
    assert m.dtype == np.float64 and np.array_equal(m, e.mean.tensor().numpy().astype(np.float64))
    assert e.mean_np is m                             # cached (the PGO caches key on its identity)
    e.mean = pp.randn_SE3()                           # as before: the cache keeps the first value
    assert e.mean_np is m
    c = copy.copy(e)                                  # conditional PGO path: a shallow copy shares the cache
    assert c.mean_np is m and c.n_frames == 3 and c.anything == "x" and c.odom_fault == 0.2
    assert abs(e.cost - float(np.linalg.norm(m[:3]))) < 1e-12
    v = VisualEdge(pp.randn_SE3(), pp.se3(torch.rand(6)), EdgeType.VISUAL, 1, 2)
    for k, val in dict(conf=0.5, noise_scale=1.0, noise_scale_along=None, noise_scale_rot=2.0, informative=True,
                       scale_corr=1.0).items():
        setattr(v, k, val)
    v2 = pickle.loads(pickle.dumps(v))
    assert (v2.from_comp_id, v2.to_comp_id, v2.conf, v2.informative) == (1, 2, 0.5, True)
    assert torch.equal(v2.mean.tensor(), v.mean.tensor())
    import weakref
    weakref.ref(v)                                    # the PGO factor caches use weak keys


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_tensor_kept_as_is():
    t = pp.randn_SE3(device="cuda")
    e = Edge(t, pp.se3(torch.rand(6, device="cuda")), EdgeType.ODOMETRY)
    assert e.mean is t
