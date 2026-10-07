"""Descriptor / spatial indexes of the keyframe database (cross/db/index.py) and locality-aware retrieval."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cross.db.index import DescriptorIndex, PCAProjection, SpatialIndex


def _unit(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)


def test_exact_index_matches_plain_matmul_and_grows_like_the_old_buffer():
    X = _unit(250, 32)
    idx = DescriptorIndex(32, device="cpu", initial_capacity=100)
    for i, x in enumerate(X):
        assert idx.add(x, 1000 + i) == i
    assert idx.n == 250 and idx.buf.shape[0] >= 350 and idx.dtype == torch.float32
    q = X[7]
    assert torch.equal(idx.scores(q), (X @ q.unsqueeze(-1)).squeeze(-1))
    rows = torch.tensor([5, 3, 200])
    assert torch.equal(idx.scores(q, rows), (X[rows] @ q.unsqueeze(-1)).squeeze(-1))
    m = idx.id_mask(max_kf_id=1100, min_kf_id=1050)
    assert m.nonzero().flatten().tolist() == list(range(50, 100))


def test_remove_moves_the_last_row():
    X = _unit(10, 8)
    idx = DescriptorIndex(8, device="cpu", initial_capacity=4)
    for i, x in enumerate(X):
        idx.add(x, i)
    assert idx.remove_row(3) == 9 and idx.n == 9 and int(idx.ids[3]) == 9
    assert torch.equal(idx.buf[3], X[9])
    assert idx.remove_row(8) is None and idx.n == 8


def test_pca_projection_keeps_cosines_of_low_rank_data_and_roundtrips(tmp_path):
    g = torch.Generator().manual_seed(1)
    basis = torch.randn(16, 256, generator=g)
    X = torch.nn.functional.normalize(torch.randn(2000, 16, generator=g) @ basis, dim=-1)
    p = PCAProjection.fit(X, 16)
    assert p.meta["explained"] > 0.99
    Z = p.apply(X[:50])
    full = X[:50] @ X[:50].T
    # centring changes cosines; ranking of neighbours is what must survive
    assert (Z @ Z.T).argsort(1)[:, -3:].eq(full.argsort(1)[:, -3:]).float().mean() > 0.9
    f = tmp_path / "pca.npz"
    p.save(str(f))
    q = PCAProjection.load(str(f))
    assert torch.allclose(q.apply(X[:5]), p.apply(X[:5]), atol=2e-3)
    idx = DescriptorIndex(256, device="cpu", initial_capacity=10, projection=q)
    for i, x in enumerate(X[:30]):
        idx.add(x, i)
    assert idx.dtype == torch.float16 and idx.buf.shape[1] == 16
    s = idx.scores(idx.encode(X[4]))
    assert int(s.argmax()) == 4 and abs(float(s[4]) - 1.0) < 2e-3


def test_ivf_backend_finds_the_exact_neighbour_most_of_the_time():
    X = _unit(6000, 32, seed=2)
    idx = DescriptorIndex(32, device="cpu", initial_capacity=100, backend="ivf", ivf_nprobe=8, ivf_min_rows=4000)
    for i, x in enumerate(X):
        idx.add(x, i)
    assert idx._ivf is not None
    hits = 0
    for i in range(0, 6000, 60):
        q = torch.nn.functional.normalize(X[i] + 0.05 * _unit(1, 32, seed=i)[0], dim=-1)
        rows = idx.ann_rows(q)
        assert rows.numel() < 6000
        hits += int(i in set(rows.tolist()))
    assert hits >= 90


def test_spatial_index_radius_queries_with_pending_rows_and_rebuild():
    sp = SpatialIndex()
    P = np.array([[0, 0, 0], [3, 0, 0], [10, 0, 0], [0, 0, 20]], float)
    sp.rebuild(np.arange(4), P, epoch=0)
    assert sp.query([[0, 0, 0]], [4.0]).tolist() == [0, 1]
    sp.add(4, [0.5, 0, 0])
    assert sp.query([[0, 0, 0], [0, 0, 20]], [1.0, 0.5]).tolist() == [0, 3, 4]
    assert sp.needs_rebuild(1, 5) and not sp.needs_rebuild(0, 5)
    sp.remap_row(4, 2)
    assert 2 in sp.query([[0.5, 0, 0]], [0.1]).tolist()


def _db_with_keyframes(n=60, d=16):
    """A KeyframeDatabase (no VPR model) with keyframes on a line, 2 m apart; embedding of image i = row i."""
    from cross.db.db import KeyframeDatabase
    from cross.db.index import DescriptorIndex
    import pypose as pp
    X = _unit(n + 5, d, seed=3)
    db = KeyframeDatabase.__new__(KeyframeDatabase)
    db.device, db.top_k = "cpu", 5
    db.score_threshold_high = db.score_threshold_low = -1.0
    db.index = DescriptorIndex(d, device="cpu", initial_capacity=10)
    db.spatial = SpatialIndex()
    db._row_kf, db._id_to_row = [], {}
    db._keyframe_by_atlas, db._index_to_atlas_idx, db._atlas_to_indices = {None: []}, {}, {None: []}
    db.vpr_model = SimpleNamespace(get_embedding=lambda i: X[i])
    for i in range(n):
        mu = pp.identity_SE3(1)
        mu.tensor()[0, 0] = 2.0 * i
        kf = SimpleNamespace(id=i, pose_mu=mu)
        r = db.index.add(X[i], i)
        db._row_kf.append(kf); db._id_to_row[i] = r
        db._keyframe_by_atlas[None].append(kf); db._index_to_atlas_idx[r] = (None, i); db._atlas_to_indices[None].append(r)
        db.spatial.add(r, [2.0 * i, 0, 0])
    return db, X


def test_query_restricted_to_rows_and_rows_near():
    db, X = _db_with_keyframes()
    rows = db.rows_near(np.array([[20.0, 0, 0]]), np.array([4.1]), epoch=0)
    assert rows.tolist() == [8, 9, 10, 11, 12]
    res = db.query(10, rows=rows, top_k=3)
    assert [k.id for k in res["keyframes"]][0] == 10 and all(8 <= k.id <= 12 for k in res["keyframes"])
    res = db.query(10, rows=rows, max_kf_id=10)
    assert all(8 <= k.id < 10 for k in res["keyframes"])


def test_locality_merge_puts_in_region_keyframes_first():
    from cross.core.config import SystemConfig
    from cross.core.system import System
    import pypose as pp
    db, X = _db_with_keyframes()
    cfg = SystemConfig()
    cfg.retrieval.locality.enabled = True
    cfg.retrieval.locality.slots = 2
    cfg.retrieval.locality.r_min = 3.0
    cfg.retrieval.locality.k_sigma = 0.0
    sysm = System.__new__(System)
    sysm.config, sysm.db = cfg, db
    sysm._locality_path, sysm._location_priors, sysm._processed_frame_num = 0.0, [], 5
    mu = pp.identity_SE3(2); mu.tensor()[0, 0] = 100.0; mu.tensor()[1, 0] = -50.0
    sysm.hypothesis_manager = SimpleNamespace(dist=(mu, pp.identity_se3(2), torch.tensor([1.0, 0.0])), pose_epoch=0)
    ranked = [(0.9, db._row_kf[3]), (0.8, db._row_kf[50]), (0.7, db._row_kf[49])]
    out = sysm._locality_merge(30, ranked)
    # region around x = 100 m (keyframes 49..51): the best two of them first, then the global list without them
    assert [k.id for _, k in out][:2] == sorted([49, 50, 51], key=lambda i: -float(X[i] @ X[30]))[:2]
    assert [k.id for _, k in out][2:] == [k.id for _, k in ranked if k.id not in {k.id for _, k in out[:2]}]
    # an external prior opens its own region; it expires after its step
    sysm.add_location_prior([0.0, 0, 0], sigma=1.0, source="gps")
    cfg.retrieval.locality.k_sigma = 1.0
    out = sysm._locality_merge(1, ranked)
    assert sysm.last_locality["sources"][-1] == "gps"
    assert out[0][1].id == 1                     # the query's own keyframe, inside the GPS region
    sysm._processed_frame_num = 10
    sysm._locality_merge(1, ranked)
    assert "gps" not in sysm.last_locality["sources"]
    # disabled: unchanged
    cfg.retrieval.locality.enabled = False
    assert sysm._locality_merge(1, ranked) is ranked
