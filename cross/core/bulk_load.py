"""Graph objects of a loaded map built from the map's columns (cross.db.store format v2) in bulk.

A map with 10^6 permanent keyframes holds ~3M keyframes and ~12M edges.  Restoring them record by record (a dict per
record, a tensor wrapper per field, a Keyframe / Edge constructor each) cost ~20 us per object (6 minutes at 10^6).
Here every field is converted once per column (numpy), and each object is created without its constructor, with
exactly the attributes the constructor and the per-record restore would give it:
- keyframe pose fields hold rows (views) of the column arrays, as the compact fields do (cross.core.types);
- edge measurements are rows of one read-only float64 copy of the column, as `pack_measurement` makes per edge;
- temporary keyframes get `normalize_SE3` over the whole column (bit-identical to one keyframe at a time).
The functions return None when a column has a form they do not handle (masked measurements, other dtypes); the
caller then restores record by record.
"""
from typing import Optional

import numpy as np
import torch

from cross.core.types import Edge, EdgeType, Keyframe, VisualEdge

_TORCH_FLOAT = {np.dtype(np.float32): torch.float32, np.dtype(np.float64): torch.float64}
_NONE = {"k": "none"}


def _ltype(col):
    import pypose as pp
    return getattr(pp, col["ltype"]) if col.get("k") == "lie" else None


def _values(col, n: int) -> list:
    """Python values of a non-tensor column, as cross.db.store._decode_column gives them (tolist already returns
    Python bools / ints / floats for the bool / int64 / float64 arrays of a "py" column)."""
    k = col.get("k")
    if k in ("tensor", "lie"):
        raise ValueError("tensor column")
    if k == "none":
        return [None] * n
    if k == "list":
        return col["v"]
    if k == "py":
        vals = col["a"].tolist()
        conv = {"bool": bool, "int": int, "float": float}[col["t"]]
        if vals and type(vals[0]) is not conv:
            vals = [conv(x) for x in vals]
    elif k == "cat":
        vals = list(map(col["cats"].__getitem__, col["a"].tolist()))
    else:
        raise ValueError(f"unknown column kind {k}")
    mask = col.get("mask")
    if mask is None:
        return vals
    out = [None] * n
    for j, i in enumerate(np.flatnonzero(mask).tolist()):
        out[i] = vals[j]
    return out


def _edge_types(col, n: int) -> list:
    """EdgeType of every row (the enum looked up once per category)."""
    if col.get("k") == "cat" and col.get("mask") is None:
        lut = [EdgeType[c] for c in col["cats"]]
        return list(map(lut.__getitem__, col["a"].tolist()))
    by_name = {}
    return [by_name.setdefault(t, EdgeType[t]) for t in _values(col, n)]


def _rows(arr: np.ndarray, mask, n: int) -> list:
    """Rows (views) of a stacked column, None where the mask says the value was None."""
    rows = list(arr)
    if mask is None:
        return rows
    out = [None] * n
    for j, i in enumerate(np.flatnonzero(mask).tolist()):
        out[i] = rows[j]
    return out


def _normalized_poses(arr: np.ndarray, chunk: int = 65536) -> np.ndarray:
    """cross.utils.lie_tensor.normalize_SE3 of every (K, 7) pose block of the column (float32, as one block each)."""
    from cross.utils.lie_tensor import normalize_SE3
    out = np.empty_like(arr)
    for s in range(0, len(arr), chunk):
        out[s:s + chunk] = normalize_SE3(torch.from_numpy(np.ascontiguousarray(arr[s:s + chunk]))).tensor().numpy()
    return out


def keyframes(enc: dict, atlas_of, normalize_mu: bool = False, images: Optional[tuple] = None,
              image_device: str = "cpu") -> Optional[list]:
    """Keyframes of an encoded record list (database keyframes or temporary keyframes), in record order.
    `atlas_of(atlas_id or None)` gives the Atlas; `images` = (index, pack) of the map's stored images (database
    keyframes)."""
    from cross.core.conditional_pose import restore
    from cross.db.store import IMAGE_FIELDS, ImageRef, _CODEC_NAMES
    n, cols = enc["n"], enc["cols"]
    if n == 0:
        return []
    col = lambda k: cols.get(k, _NONE)                                          # noqa: E731
    fields = {}
    for name in ("pose_mu", "pose_std", "pose_weights", "pose_charts"):
        c = col(name)
        if c["k"] == "none":
            fields[name] = ([None] * n, None)
        elif c["k"] in ("tensor", "lie"):
            a = c["a"]
            if name == "pose_mu" and normalize_mu:
                a = _normalized_poses(a)
            fields[name] = (_rows(a, c.get("mask"), n), _ltype(c))
        else:
            return None
    ids = _values(col("id"), n)
    atlas_ids = _values(col("atlas_id"), n)
    timestamps = _values(col("timestamp"), n)
    temporary = _values(col("temporary"), n)
    last_pgo = _values(col("last_pgo_step"), n)
    metric = _values(col("metric_source"), n)
    cond = _values(col("conditional_poses"), n)
    imgs = {f: [None] * n for f in ("raw_rgb_image", "depth_image", "raw_rgb_right")}
    if images is not None:
        idx, pack = images
        names = idx["dtype_names"]
        shapes = [tuple(r[:d]) for r, d in zip(idx["shape"].tolist(), idx["ndim"].tolist())]
        for row, f, off, ln, cod, shp, dt in zip(idx["row"].tolist(), idx["field"].tolist(), idx["offset"].tolist(),
                                                 idx["length"].tolist(), idx["codec"].tolist(), shapes,
                                                 idx["dtype"].tolist()):
            ref = ImageRef.__new__(ImageRef)
            ref.pack, ref.offset, ref.length, ref.codec = pack, off, ln, _CODEC_NAMES[cod]
            ref.shape, ref.dtype, ref.device = shp, names[dt], str(image_device)
            imgs[IMAGE_FIELDS[f]][row] = ref
    atlases = {a: atlas_of(a) for a in set(atlas_ids)}
    mu, mu_lt = fields["pose_mu"]
    sd, sd_lt = fields["pose_std"]
    w, w_lt = fields["pose_weights"]
    ch, ch_lt = fields["pose_charts"]
    rgb, dep, right = imgs["raw_rgb_image"], imgs["depth_image"], imgs["raw_rgb_right"]
    new = Keyframe.__new__
    out = []
    append = out.append
    for i in range(n):
        kf = new(Keyframe)
        # the attributes, in the order, that Keyframe(...) followed by `kf.id = saved id` gives (the compact fields'
        # storage names: cross.core.types._TensorField / _ImageField)
        kf._pose_mu = mu[i]
        kf._pose_mu_lt = mu_lt if mu[i] is not None else None
        kf._pose_std = sd[i]
        kf._pose_std_lt = sd_lt if sd[i] is not None else None
        kf._pose_weights = w[i]
        kf._pose_weights_lt = w_lt if w[i] is not None else None
        kf._raw_rgb_image, kf._depth_image, kf._raw_rgb_right = rgb[i], dep[i], right[i]
        kf.atlas, kf.timestamp, kf.temporary, kf.last_pgo_step = atlases[atlas_ids[i]], timestamps[i], temporary[i], last_pgo[i]
        kf._pose_charts = ch[i]
        kf._pose_charts_lt = ch_lt if ch[i] is not None else None
        kf.metric_source = metric[i]
        kf.conditional_poses = restore(cond[i]) if cond[i] is not None else None
        kf.id = ids[i]
        append(kf)
    Keyframe._next_id += n                  # as n constructions would (System.load_map sets it afterwards)
    return out


def _measurement(col):
    """(rows, ltype, torch dtype) of an edge measurement column, as pack_measurement gives per edge; None if the
    column has another form."""
    if col.get("k") not in ("tensor", "lie") or col.get("mask") is not None:
        return None
    a = col["a"]
    dt = _TORCH_FLOAT.get(a.dtype)
    if dt is None:
        return None
    m = a.astype(np.float64)
    m.flags.writeable = False
    return list(m), _ltype(col), dt


def edges(enc: dict, visual: bool) -> Optional[list]:
    """Edges of an encoded record list (odometry edge records, or the flattened visual edge records)."""
    from cross.core.conditional_pose import ConditionalPose
    n, cols = enc["n"], enc["cols"]
    if n == 0:
        return []
    col = lambda k: cols.get(k, _NONE)                                          # noqa: E731
    mean, std = _measurement(col("mean")), _measurement(col("std"))
    if mean is None or std is None:
        return None
    m, m_lt, m_dt = mean
    s, s_lt, s_dt = std
    types = _edge_types(col("type"), n)
    cond = _values(col("conditional_pose"), n)
    out = []
    append = out.append
    if not visual:
        n_frames = _values(col("n_frames"), n)
        fault = _values(col("odom_fault"), n)
        new = Edge.__new__
        for i in range(n):
            e = new(Edge)
            e._m, e._mt, e._md = m[i], m_lt, m_dt
            e._s, e._st, e._sd = s[i], s_lt, s_dt
            e.type, e._cost, e.conditional_pose, e.n_frames, e.conf = types[i], None, None, n_frames[i], None
            if fault[i]:
                e.odom_fault = float(fault[i])
            if cond[i] is not None:
                e.conditional_pose = ConditionalPose.from_record(cond[i])
            append(e)
        return out
    fc, tc = _values(col("from_comp_id"), n), _values(col("to_comp_id"), n)
    conf, ns = _values(col("conf"), n), _values(col("noise_scale"), n)
    nsa, nsr = _values(col("noise_scale_along"), n), _values(col("noise_scale_rot"), n)
    inf = _values(col("informative"), n)
    new = VisualEdge.__new__
    for i in range(n):
        e = new(VisualEdge)
        e._m, e._mt, e._md = m[i], m_lt, m_dt
        e._s, e._st, e._sd = s[i], s_lt, s_dt
        e.type, e._cost, e.conditional_pose, e.n_frames = types[i], None, None, None
        e.from_comp_id, e.to_comp_id = fc[i], tc[i]
        e.conf, e.noise_scale, e.noise_scale_along, e.noise_scale_rot, e.informative = conf[i], ns[i], nsa[i], nsr[i], inf[i]
        if cond[i] is not None:
            e.conditional_pose = ConditionalPose.from_record(cond[i])
        append(e)
    return out


def keys(enc) -> list:
    """Dict keys of a map's encoded key column (ints or tuples of ints)."""
    from cross.db.store import _decode_keys
    if enc["k"] == "tuple":
        return list(map(tuple, enc["a"].tolist()))
    if enc["k"] == "int":
        return enc["a"].tolist()
    return _decode_keys(enc)


def grouped(items: list, counts: np.ndarray) -> list:
    """Consecutive slices of `items` of the given lengths."""
    ends = np.cumsum(counts).tolist()
    starts = [0] + ends[:-1]
    return [items[a:b] for a, b in zip(starts, ends)]


# --------------------------------------------------------------------------------------------------------------------
# Saving: the encoded columns of cross.db.store (encode_records / encode_dict_of_records / encode_dict_of_lists of the
# save_state records) built from the graph objects directly.  The records would hold one tensor per field (a LieTensor
# `.cpu()` costs ~20 us: a 10^5-keyframe map took ~90 s to serialize).  Each function returns exactly what the record
# path encodes, or None for a form it does not handle (the caller then builds the records).
# --------------------------------------------------------------------------------------------------------------------
_NP_OK = {np.dtype(t) for t in (np.float16, np.float32, np.float64, np.uint8, np.int8, np.int16, np.int32, np.int64,
                                np.bool_)}


def _tensor_column(arrs: list, lts: list, n: int):
    """store._encode_column of the tensors held compactly as (array, ltype) pairs; None if any is not a compact array."""
    from cross.db.store import _LTYPES
    present = [i for i, a in enumerate(arrs) if a is not None]
    if not present:
        return {"k": "none", "n": n}
    mask = None if len(present) == n else np.array([a is not None for a in arrs], dtype=bool)
    a0, lt0 = arrs[present[0]], lts[present[0]]
    if type(a0) is not np.ndarray or a0.dtype not in _NP_OK:
        return None
    name = None
    if lt0 is not None:
        name = _LTYPES.get(type(lt0).__name__)
        if name is None:
            return None
    dt, sh, lt_cls = a0.dtype, a0.shape, type(lt0)
    for i in present:
        a, lt = arrs[i], lts[i]
        if type(a) is not np.ndarray or a.dtype != dt or a.shape != sh or type(lt) is not lt_cls:
            return None
    vals = [arrs[i] for i in present]
    return {"k": "lie" if name is not None else "tensor", "ltype": name, "mask": mask, "a": np.stack(vals)}


def _measurement_column(edges: list, which: str):
    """The column of edge.mean / edge.std (`which` "m" / "s") as the records' tensors encode: the float64 stored
    values converted back to the measurement's dtype; None if an edge holds its measurement in another form."""
    from cross.core.types import _EDGE_FLOATS
    from cross.db.store import _LTYPES
    if not edges:
        return {"k": "none", "n": 0}
    a_attr, lt_attr, dt_attr = "_" + which, "_" + which + "t", "_" + which + "d"
    e0 = edges[0]
    a0, lt0, dt0 = getattr(e0, a_attr), getattr(e0, lt_attr), getattr(e0, dt_attr)
    if type(a0) is not np.ndarray or dt0 not in _EDGE_FLOATS:
        return None
    name = None
    if lt0 is not None:
        name = _LTYPES.get(type(lt0).__name__)
        if name is None:
            return None
    sh, lt_cls = a0.shape, type(lt0)
    rows = []
    for e in edges:
        a = getattr(e, a_attr)
        if type(a) is not np.ndarray or a.shape != sh or getattr(e, dt_attr) is not dt0 or \
                type(getattr(e, lt_attr)) is not lt_cls:
            return None
        rows.append(a)
    arr = np.stack(rows).astype({torch.float32: np.float32, torch.float64: np.float64}[dt0])
    return {"k": "lie" if name is not None else "tensor", "ltype": name, "mask": None, "a": arr}


def _records_columns(keys: list, cols: dict, n: int) -> dict:
    return {"__records__": True, "n": n, "keys": list(keys), "cols": {k: cols[k] for k in keys}}


def encode_keyframes(kfs: list, image_fields: bool) -> Optional[dict]:
    """encode_records of the keyframe records of HypothesisManager.save_state (temporary keyframes) or, with
    `image_fields`, of KeyframeDatabase.save_state with its image fields already set to None (store.write_map encodes
    the images separately)."""
    from cross.core.conditional_pose import records
    from cross.db.store import _encode_column
    n = len(kfs)
    if n == 0:
        return None
    cols = {}
    for f in ("pose_mu", "pose_std", "pose_weights", "pose_charts"):
        d = [kf.__dict__ for kf in kfs]
        c = _tensor_column([x.get("_" + f) for x in d], [x.get("_" + f + "_lt") for x in d], n)
        if c is None:
            return None
        cols[f] = c
    cols["id"] = _encode_column([kf.id for kf in kfs])
    cols["metric_source"] = _encode_column([kf.metric_source for kf in kfs])
    cols["conditional_poses"] = _encode_column([records(kf.conditional_poses) for kf in kfs])
    cols["timestamp"] = _encode_column([kf.timestamp for kf in kfs])
    cols["temporary"] = _encode_column([kf.temporary for kf in kfs])
    cols["atlas_id"] = _encode_column([kf.atlas.id if kf.atlas is not None else None for kf in kfs])
    if image_fields:
        for f in ("raw_rgb_image", "depth_image", "raw_rgb_right"):
            cols[f] = {"k": "none", "n": n}
        cols["last_pgo_step"] = _encode_column([kf.last_pgo_step for kf in kfs])
        keys = ["id", "raw_rgb_image", "depth_image", "raw_rgb_right", "pose_mu", "pose_std", "pose_weights",
                "pose_charts", "metric_source", "conditional_poses", "atlas_id", "timestamp", "temporary",
                "last_pgo_step"]
    else:
        cols["last_pgo_step"] = _encode_column([getattr(kf, "last_pgo_step", -1) for kf in kfs])
        keys = ["id", "pose_mu", "pose_std", "pose_weights", "pose_charts", "metric_source", "conditional_poses",
                "timestamp", "temporary", "atlas_id", "last_pgo_step"]
    return _records_columns(keys, cols, n)


def encode_edges(edges: list, visual: bool) -> Optional[dict]:
    """encode_records of the edge records of HypothesisManager.save_state (odometry, or visual edges flattened)."""
    from cross.db.store import _encode_column
    n = len(edges)
    if n == 0:
        return None
    if not visual and any(getattr(e, "odom_fault", None) for e in edges):
        return None                      # those records carry one more key: the record path stores them as a list
    mean, std = _measurement_column(edges, "m"), _measurement_column(edges, "s")
    if mean is None or std is None:
        return None
    cols = {"mean": mean, "std": std, "type": _encode_column([e.type.name for e in edges]),
            "conditional_pose": _encode_column([e.conditional_pose.record() if e.conditional_pose is not None
                                                else None for e in edges])}
    if not visual:
        cols["n_frames"] = _encode_column([getattr(e, "n_frames", None) for e in edges])
        return _records_columns(["mean", "std", "type", "conditional_pose", "n_frames"], cols, n)
    cols["from_comp_id"] = _encode_column([e.from_comp_id for e in edges])
    cols["to_comp_id"] = _encode_column([e.to_comp_id for e in edges])
    for k in ("conf", "noise_scale", "noise_scale_along", "noise_scale_rot", "informative"):
        cols[k] = _encode_column([getattr(e, k, None) for e in edges])
    return _records_columns(["mean", "std", "type", "from_comp_id", "to_comp_id", "conditional_pose", "conf",
                             "noise_scale", "noise_scale_along", "noise_scale_rot", "informative"], cols, n)
