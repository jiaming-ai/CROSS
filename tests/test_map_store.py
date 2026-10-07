"""Map storage format v2 (cross/db/store.py): exact round trip, lazy images, old pickles, symlinks, incremental saves,
the live spool."""
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pp = pytest.importorskip("pypose")
pytest.importorskip("cv2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cross.core.config import HypothesisConfig, StorageConfig, SystemConfig  # noqa: E402
from cross.core.hypothesis import HypothesisManager  # noqa: E402
from cross.core.types import Edge, EdgeType, Keyframe, VisualEdge  # noqa: E402
from cross.db import store  # noqa: E402
from convert_map import same  # noqa: E402


def _rng_image(rng, h=48, w=64):
    # smooth content (compressible) plus noise
    y, x = np.mgrid[:h, :w]
    base = (np.sin(x / 7.0)[None] * 60 + np.cos(y / 5.0)[None] * 60 + 128).repeat(3, 0)
    return torch.from_numpy(np.clip(base + rng.normal(0, 8, (3, h, w)), 0, 255).astype(np.uint8))


def _db(monkeypatch, scfg=None):
    import cross.db.db as db_module
    monkeypatch.setattr(db_module, "BoQ", lambda **_: SimpleNamespace(
        get_embed_dim=lambda: 8, get_embedding=lambda img: torch.as_tensor(img).float().mean() * torch.ones(8)))
    system = SimpleNamespace(config=SystemConfig(storage=scfg or StorageConfig(encode_ahead=False)))
    return db_module.KeyframeDatabase(system, device="cpu")


def _state(monkeypatch, n=6, depth=True, right=True, scfg=None):
    """A saved-map dict (as System.save_map builds it) with n permanent and n temporary keyframes."""
    rng = np.random.default_rng(0)
    db = _db(monkeypatch, scfg)
    atlas = db.create_atlas()
    system = SimpleNamespace(device="cpu", topo_map=None, loaded_node_ids=frozenset(), config=SystemConfig())
    hm = HypothesisManager(system, 3, HypothesisConfig())
    hm.dist = (pp.identity_SE3(3), pp.se3(torch.ones(3, 6) * .1), torch.tensor([1., 0., 0.]))
    hm.create_hypothesis_branch(0, 0)
    prev = None
    for i in range(n):
        mu = pp.randn_SE3(3)
        kf = db.insert(i, _rng_image(rng), torch.from_numpy((rng.random((1, 48, 64)) * 5).astype(np.float16)) if depth else None,
                       mu=mu, sigma=pp.se3(torch.rand(3, 6)), weights=torch.tensor([1., 0., 0.]), atlas=atlas,
                       timestamp=float(i), raw_rgb_right=_rng_image(rng) if right else None,
                       metric_source=None if i % 2 else dict(source_id=f"s{i}"))
        hm.nodes[kf.id] = kf
        t = Keyframe(pp.randn_SE3(3), pp.se3(torch.rand(3, 6)), torch.tensor([1., 0., 0.]), timestamp=i + .5,
                     temporary=True)
        hm.nodes[t.id] = t
        e = Edge(pp.randn_SE3(), pp.se3(torch.rand(6)), EdgeType.ODOMETRY)
        e.n_frames = i + 1
        hm.odom_edges[(kf.id, t.id)] = e
        if prev is not None:
            ve = VisualEdge(pp.randn_SE3(), pp.se3(torch.rand(6)), from_comp_id=0, to_comp_id=0)
            ve.conf = 0.5 if i % 3 else None
            ve.informative = bool(i % 2)
            hm.hypotheses[0].visual_edges.setdefault((prev, kf.id), []).append(ve)
            hm.hypotheses[0].visual_adjacency.setdefault(kf.id, set()).add(prev)
        prev = kf.id
    data = {"config": {"a": 1}, "db_data": db.save_state(), "hypo_data": hm.save_state(),
            "class_vars": {"keyframe_next_id": Keyframe._next_id}, "current_atlas_id": 0}
    return data, db


def test_exact_round_trip(tmp_path, monkeypatch):
    data, _ = _state(monkeypatch)
    legacy = store.materialize(data)
    st = store.write_map(tmp_path / "map.pkl", data, StorageConfig(image_codec="png", descriptor_dtype="float32"))
    assert st["encoded"] == 6 * 3
    back = store.read_map(tmp_path / "map.pkl")
    assert store.is_ref(back["db_data"]["keyframes"][0]["raw_rgb_image"])
    assert same(legacy, back) == []
    for codec in ("webp_lossless", "raw"):
        store.write_map(tmp_path / codec / "map.pkl", data, StorageConfig(image_codec=codec, descriptor_dtype="float32"))
        assert same(legacy, store.read_map(tmp_path / codec / "map.pkl")) == []


def test_old_pickle_still_loads(tmp_path, monkeypatch):
    data, _ = _state(monkeypatch)
    legacy = store.materialize(data)
    with open(tmp_path / "map.pkl", "wb") as f:
        pickle.dump(legacy, f)
    back = store.read_map(tmp_path / "map.pkl")
    assert same(legacy, back) == []
    assert torch.is_tensor(back["db_data"]["keyframes"][0]["raw_rgb_image"])


def test_lazy_keyframe_images_and_db_load(tmp_path, monkeypatch):
    data, db = _state(monkeypatch)
    store.write_map(tmp_path / "map.pkl", data, StorageConfig())
    back = store.read_map(tmp_path / "map.pkl")
    db2 = _db(monkeypatch)
    kfs = db2.load_state(back["db_data"], "cpu")
    orig = {k.id: k for k in db.get_all_keyframes()}
    store.set_decode_cache(4)
    for kid, kf in kfs.items():
        assert store.is_ref(kf.stored_image("raw_rgb_image"))            # nothing decoded at load
        assert kf.has_image("depth_image") and kf.has_image("raw_rgb_right")
        assert torch.equal(kf.raw_rgb_image, orig[kid].raw_rgb_image)    # decoded on access, exact
        assert torch.equal(kf.depth_image, orig[kid].depth_image) and kf.depth_image.dtype == torch.float16
        assert torch.equal(kf.raw_rgb_right, orig[kid].raw_rgb_right)
    assert store.decode_cache_stats()["size"] <= 4
    store.set_decode_cache(1024)


def test_symlinked_map_finds_its_data(tmp_path, monkeypatch):
    data, _ = _state(monkeypatch)
    (tmp_path / "maps").mkdir()
    (tmp_path / "query").mkdir()
    store.write_map(tmp_path / "maps" / "map.pkl", data, StorageConfig(descriptor_dtype="float32"))
    os.symlink(tmp_path / "maps" / "map.pkl", tmp_path / "query" / "map.pkl")
    back = store.read_map(tmp_path / "query" / "map.pkl")
    assert same(store.materialize(data), back) == []
    assert not (tmp_path / "query" / "map.pkl.store").exists()


def test_incremental_save_appends_only_new_images(tmp_path, monkeypatch):
    data, _ = _state(monkeypatch, n=4)
    p = tmp_path / "map.pkl"
    store.write_map(p, data, StorageConfig(descriptor_dtype="float32"))
    back = store.read_map(p)
    size0 = store.sidecar_dir(p).joinpath(next(store.sidecar_dir(p).glob("images-*.pack")).name).stat().st_size
    rec = dict(back["db_data"]["keyframes"][-1])
    rec["id"] = 999
    rec["raw_rgb_image"] = _rng_image(np.random.default_rng(5))
    back["db_data"]["keyframes"].append(rec)
    st = store.write_map(p, back, StorageConfig(descriptor_dtype="float32"))
    assert st["kept"] == 4 * 3 + 2 and st["encoded"] == 1 and st["copied"] == 0
    packs = list(store.sidecar_dir(p).glob("images-*.pack"))
    assert len(packs) == 1 and packs[0].stat().st_size > size0
    again = store.read_map(p)
    assert torch.equal(again["db_data"]["keyframes"][-1]["raw_rgb_image"].load(), rec["raw_rgb_image"])
    assert same(store.materialize(back), again) == []


def test_overwriting_a_map_removes_the_old_pack(tmp_path, monkeypatch):
    data, _ = _state(monkeypatch, n=3)
    p = tmp_path / "map.pkl"
    store.write_map(p, data, StorageConfig())
    old = list(store.sidecar_dir(p).glob("images-*.pack"))
    store.write_map(p, store.materialize(data), StorageConfig())       # a different map (tensors, no references)
    new = list(store.sidecar_dir(p).glob("images-*.pack"))
    assert len(new) == 1 and new != old
    store.remove_map(p)
    assert not p.exists() and not store.sidecar_dir(tmp_path / "map.pkl").exists()


def test_copy_to_another_map_without_reencoding(tmp_path, monkeypatch):
    data, _ = _state(monkeypatch, n=3)
    cfg = StorageConfig(image_codec="jpeg", image_quality=80, descriptor_dtype="float32")
    store.write_map(tmp_path / "a" / "map.pkl", data, cfg)
    a = store.read_map(tmp_path / "a" / "map.pkl")
    st = store.write_map(tmp_path / "b" / "map.pkl", a, cfg)
    assert st["copied"] == 3 * 3 and st["encoded"] == 0
    assert same(store.materialize(a), store.read_map(tmp_path / "b" / "map.pkl")) == []


def test_lossy_codecs_keep_shape_and_dtype(tmp_path, monkeypatch):
    data, _ = _state(monkeypatch, n=2)
    for codec in ("jpeg", "webp"):
        store.write_map(tmp_path / codec / "map.pkl", data, StorageConfig(image_codec=codec, image_quality=90))
        back = store.read_map(tmp_path / codec / "map.pkl")
        for r0, r1 in zip(data["db_data"]["keyframes"], back["db_data"]["keyframes"]):
            a, b = r0["raw_rgb_image"], r1["raw_rgb_image"].load()
            assert a.shape == b.shape and b.dtype == torch.uint8
            assert (a.float() - b.float()).abs().mean() < 10           # noisy synthetic image (real keyframes: ~1)
            assert torch.equal(r0["depth_image"], r1["depth_image"].load())   # depth stays exact


def test_fp16_descriptors(tmp_path, monkeypatch):
    data, _ = _state(monkeypatch, n=3)
    store.write_map(tmp_path / "map.pkl", data, StorageConfig(descriptor_dtype="float16"))
    back = store.read_map(tmp_path / "map.pkl")
    e0, e1 = data["db_data"]["embeddings"], back["db_data"]["embeddings"]
    assert e1.dtype == e0.dtype and torch.allclose(e0, e1, rtol=1e-3)


def test_live_spool_keeps_newest_images_in_memory(monkeypatch, tmp_path):
    scfg = StorageConfig(max_ram_images=2, spill_dir=str(tmp_path / "spool"), image_codec="png", encode_ahead=False)
    data, db = _state(monkeypatch, n=5, scfg=scfg)
    db._spool.flush()
    kfs = db.get_all_keyframes()
    held = [torch.is_tensor(k.stored_image("raw_rgb_image")) for k in kfs]
    assert held == [False, False, False, True, True]
    data2, db2 = _state(monkeypatch, n=5)                                  # same content, all in memory
    for a, b in zip(kfs, db2.get_all_keyframes()):
        assert torch.equal(a.raw_rgb_image, b.raw_rgb_image) and torch.equal(a.depth_image, b.depth_image)
    # every keyframe was encoded as it arrived (evicted or not): saving copies bytes, encodes nothing
    st = store.write_map(tmp_path / "m" / "map.pkl", dict(data, db_data=db.save_state()), scfg)
    assert st["copied"] == 5 * 3 and st["encoded"] == 0


def test_columns_with_missing_values():
    recs = [{"a": 1, "b": None, "c": torch.ones(2), "d": "x"}, {"a": 2, "b": 0.5, "c": None, "d": "y"},
            {"a": 3, "b": None, "c": torch.zeros(2), "d": ("odd", 1)}]
    enc = store.encode_records(recs)
    assert enc["cols"]["a"]["k"] == "py" and enc["cols"]["c"]["k"] == "tensor" and enc["cols"]["d"]["k"] == "list"
    assert same(recs, store.decode_records(pickle.loads(pickle.dumps(enc)))) == []


def test_spooled_images_are_repointed_to_the_saved_map(monkeypatch, tmp_path):
    """System.save_map's callback: spooled images move into the map's pack once, later saves keep them."""
    scfg = StorageConfig(max_ram_images=2, spill_dir=str(tmp_path / "spool"), image_codec="png", encode_ahead=False)
    data, db = _state(monkeypatch, n=5, scfg=scfg)
    p = tmp_path / "m" / "map.pkl"

    def save():
        state = db.save_state()
        rows = [r["id"] for r in state["keyframes"]]
        by_id = {k.id: k for k in db.get_all_keyframes()}

        def on_written(row, field, ref):                     # as System.save_map
            kf = by_id[rows[row]]
            old = kf.stored_image(field)
            if store.is_ref(old) and old.pack.uid != ref.pack.uid:
                setattr(kf, field, ref.to(old.device))
            pre = kf.__dict__.get("_pre_" + field)
            if pre is not None and pre.pack.uid != ref.pack.uid:
                kf.__dict__["_pre_" + field] = ref.to(pre.device)
        st = store.write_map(p, dict(data, db_data=state), scfg, on_written=on_written)
        db._spool.retarget(st["pack"])
        return st

    st1 = save()
    assert st1["copied"] == 15 and st1["encoded"] == 0
    kf_new = db.insert(99, _rng_image(np.random.default_rng(9)), None, mu=pp.randn_SE3(3), sigma=pp.se3(torch.rand(3, 6)),
                       weights=torch.tensor([1., 0., 0.]), atlas=db.get_all_atlases()[0])
    db._spool.flush()                                                        # one more keyframe spilled into the map
    st2 = save()                                  # re-pointed images and the new one (spilled into the map) kept
    assert st2["copied"] == 0 and st2["encoded"] == 0 and st2["kept"] == 15 + 1
    back = store.read_map(p)
    assert torch.equal(back["db_data"]["keyframes"][-1]["raw_rgb_image"].load(), kf_new.raw_rgb_image)


def test_depth_drop_bits_bounds_the_relative_error():
    rng = np.random.default_rng(1)
    d = (rng.random((1, 40, 50)) * 30 + 0.01).astype(np.float16)
    d[0, :3] = 0
    for b in (2, 3, 4):
        back = store.decode_array(store.encode_array(d, "png16", depth_drop_bits=b), "png16", d.shape, "float16")
        x, y = d.astype(np.float32), back.astype(np.float32)
        m = x > 0
        assert np.all(y[~m] == 0) and np.max(np.abs(y[m] - x[m]) / x[m]) <= 2.0 ** (b - 11) + 1e-6


def test_encode_ahead_saves_by_copying(monkeypatch, tmp_path):
    """encode_ahead: images are encoded as keyframes arrive (tensors stay); a save copies bytes, the map is the same."""
    scfg = StorageConfig(spill_dir=str(tmp_path / "spool"), image_codec="png")            # encode_ahead on by default
    data, db = _state(monkeypatch, n=4, scfg=scfg)
    db._spool.flush()
    assert all(torch.is_tensor(k.stored_image("raw_rgb_image")) for k in db.get_all_keyframes())
    data_ref, _ = _state(monkeypatch, n=4)                                                  # same content, nothing pre-encoded
    st = store.write_map(tmp_path / "a" / "map.pkl", dict(data, db_data=db.save_state()), scfg)
    assert st["encoded"] == 0 and st["copied"] == 4 * 3
    back = store.read_map(tmp_path / "a" / "map.pkl")
    ref = store.materialize(data_ref)
    for r0, r1 in zip(ref["db_data"]["keyframes"], back["db_data"]["keyframes"]):
        for f in store.IMAGE_FIELDS:
            assert torch.equal(r0[f], r1[f].load())
