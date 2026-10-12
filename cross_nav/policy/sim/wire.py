"""Messages between the simulator workers and the policy server.

The two sides run different Python/numpy versions, so arrays travel as (dtype, shape, bytes), never as pickled
ndarrays (numpy 2 pickles do not load in numpy 1).
"""
from __future__ import annotations

import pickle

import numpy as np


def _pack(x):
    if isinstance(x, np.ndarray):
        return ('__nd__', x.dtype.str, x.shape, np.ascontiguousarray(x).tobytes())
    if isinstance(x, dict):
        return {k: _pack(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_pack(v) for v in x)
    if isinstance(x, np.generic):
        return x.item()
    return x


def _unpack(x):
    if isinstance(x, tuple) and len(x) == 4 and x[0] == '__nd__':
        return np.frombuffer(x[3], dtype=np.dtype(x[1])).reshape(x[2]).copy()
    if isinstance(x, dict):
        return {k: _unpack(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_unpack(v) for v in x)
    return x


def send(conn, msg):
    conn.send_bytes(pickle.dumps(_pack(msg), protocol=4))


def recv(conn):
    return _unpack(pickle.loads(conn.recv_bytes()))
