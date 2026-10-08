"""The bulk restore of a v2 map (cross.core.bulk_load, System.load_map) gives exactly the graph objects of the
record-by-record restore: every attribute, value bit and dict order."""
from types import SimpleNamespace

import numpy as np
import pypose as pp
import torch

from cross.core import bulk_load
from cross.core.config import HypothesisConfig, StorageConfig, SystemConfig
from cross.core.hypothesis import HypothesisManager
from cross.core.types import Edge, Keyframe, VisualEdge
from cross.db import store

from test_map_store import _db, _state


def _rich_state(monkeypatch):
    """_state plus the forms the bulk path must reproduce: non-unit quaternions of temporary keyframes (normalized at
    load), an odometry fault, an empty visual-edge list, a missing chart column, noise-scale metadata."""
    data, db = _state(monkeypatch, n=7)
    hy = data["hypo_data"]
    for k, t in enumerate(hy["temp_keyframes"]):
        mu = t["pose_mu"].tensor().clone()
        mu[:, 3:7] *= 0.97 + 0.01 * k
        t["pose_mu"] = pp.SE3(mu)
        t["pose_charts"] = torch.tensor([k, 0, 1]) if k % 2 else None
    for j, (key, e) in enumerate(hy["odom_edges"].items()):
        e["odom_fault"] = 0.25 if j == 2 else None
    h0 = hy["hypotheses_data"][0]
    first = next(iter(h0["visual_edges"]))
    h0["visual_edges"][(first[1], first[0])] = []              # an empty list (a removed edge) is not restored
    for j, l in enumerate(h0["visual_edges"].values()):
        for e in l:
            e["noise_scale"], e["noise_scale_along"], e["noise_scale_rot"] = 1.5, (0.5 if j % 2 else None), 2.0
    return data


def _restore(d, monkeypatch, bulk: bool):
    if not bulk:
        monkeypatch.setattr(bulk_load, "keyframes", lambda *a, **k: None)
        monkeypatch.setattr(bulk_load, "edges", lambda *a, **k: None)
    db = _db(monkeypatch)
    n0 = Keyframe._next_id
    existing = db.load_state(d["db_data"], "cpu", pose_device="cpu")
    system = SimpleNamespace(device="cpu", topo_map=None, loaded_node_ids=frozenset(), config=SystemConfig())
    hm = HypothesisManager(system, 3, HypothesisConfig())
    hm.load_state(d["hypo_data"], db, "cpu", "cpu", existing)
    monkeypatch.undo()
    return db, hm, Keyframe._next_id - n0


def _same(x, y, where):
    if isinstance(x, np.ndarray):
        assert isinstance(y, np.ndarray), where
        assert x.dtype == y.dtype and x.shape == y.shape and x.flags.writeable == y.flags.writeable, where
        assert np.array_equal(x.view(np.uint8), y.view(np.uint8)), where
    elif store.is_ref(x):
        assert x.__getstate__() == y.__getstate__(), where
    elif torch.is_tensor(x):
        assert type(x) is type(y) and torch.equal(x, y), where
    else:
        assert x is y or x == y, (where, x, y)


def _slots(cls):
    out = []
    for c in cls.__mro__:
        out += [s for s in getattr(c, "__slots__", ()) if s not in ("__dict__", "__weakref__")]
    return out


def test_bulk_restore_equals_record_restore(tmp_path, monkeypatch):
    data = _rich_state(monkeypatch)
    store.write_map(tmp_path / "map.pkl", data, StorageConfig(image_codec="png", descriptor_dtype="float32"))
    a_db, a, na = _restore(store.read_map(tmp_path / "map.pkl", packed=True), monkeypatch, bulk=False)
    b_db, b, nb = _restore(store.read_map(tmp_path / "map.pkl", packed=True), monkeypatch, bulk=True)
    assert na == nb
    assert list(a.nodes) == list(b.nodes)
    for k in a.nodes:
        x, y = vars(a.nodes[k]), vars(b.nodes[k])
        assert list(x) == list(y), k
        for f in x:
            _same(x[f], y[f], (k, f))
    assert list(a.odom_edges) == list(b.odom_edges)
    va, vb = a.hypotheses[0].visual_edges, b.hypotheses[0].visual_edges
    assert list(va) == list(vb) and all(len(va[k]) == len(vb[k]) for k in va)
    pairs = [(a.odom_edges[k], b.odom_edges[k]) for k in a.odom_edges] + \
            [(x, y) for k in va for x, y in zip(va[k], vb[k])]
    for x, y in pairs:
        assert type(x) is type(y)
        for s in _slots(type(x)):
            assert hasattr(x, s) == hasattr(y, s), s
            if hasattr(x, s):
                _same(getattr(x, s), getattr(y, s), s)
        assert vars(x) == vars(y)
        assert np.array_equal(x.mean_np, y.mean_np) and torch.equal(x.mean.tensor(), y.mean.tensor())
    assert a.hypotheses[0].visual_adjacency == b.hypotheses[0].visual_adjacency
    assert list(a.hypotheses[0].visual_adjacency) == list(b.hypotheses[0].visual_adjacency)
    assert a_db._row_kf == b_db._row_kf and a_db._id_to_row == b_db._id_to_row
    assert {k: (v[0].id, v[1]) for k, v in a_db._index_to_atlas_idx.items()} == \
           {k: (v[0].id, v[1]) for k, v in b_db._index_to_atlas_idx.items()}
    # the temporary keyframes' quaternions were normalized as the record path does
    t = next(kf for kf in b.nodes.values() if kf.temporary)
    q = t.pose_mu.tensor()[:, 3:7]
    assert torch.allclose(q.norm(dim=-1), torch.ones(3))


def test_bulk_keyframes_write_through(tmp_path, monkeypatch):
    """A bulk-restored keyframe's pose field is a row view: in-place writes land in that keyframe only."""
    data = _rich_state(monkeypatch)
    store.write_map(tmp_path / "map.pkl", data, StorageConfig(image_codec="png", descriptor_dtype="float32"))
    _, hm, _ = _restore(store.read_map(tmp_path / "map.pkl", packed=True), monkeypatch, bulk=True)
    ks = [k for k, kf in hm.nodes.items() if not kf.temporary][:2]
    before = hm.nodes[ks[1]].pose_mu.tensor().clone()
    hm.nodes[ks[0]].pose_mu[0] = pp.identity_SE3()
    assert torch.equal(hm.nodes[ks[0]].pose_mu.tensor()[0], pp.identity_SE3().tensor())
    assert torch.equal(hm.nodes[ks[1]].pose_mu.tensor(), before)
