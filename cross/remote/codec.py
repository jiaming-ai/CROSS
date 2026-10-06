"""The wire format of a remote session (cross.remote.grpc_link): a message (nested dicts / lists of numbers, strings,
None and numpy arrays; images as JPEG or PNG) as one bytes payload, a JSON header and the binary parts after it.  No
pickle: a payload can only produce these types."""

import json
import struct

import numpy as np

IMAGE_KEYS = ("rgb", "rgb_right")


def _image_bytes(img, quality):
    import cv2
    img = np.ascontiguousarray(np.asarray(img)[..., ::-1])
    if quality:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    else:
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise ValueError("image encoding failed")
    return buf.tobytes()


def _image(data):
    import cv2
    out = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return np.ascontiguousarray(out[..., ::-1])


def encode(msg, jpeg: int = 0) -> bytes:
    """A message as bytes; the images under IMAGE_KEYS (H x W x 3 uint8 RGB) as JPEG at this quality, or lossless PNG."""
    blobs = []

    def enc(x, key=None):
        if x is None or isinstance(x, (bool, str)):
            return x
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, (float, np.floating)):
            v = float(x)
            return v if np.isfinite(v) else {"__f__": repr(v)}
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, dict):
            return {str(k): enc(v, k) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [enc(v) for v in x]
        if hasattr(x, "detach"):                         # a torch tensor
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            if key in IMAGE_KEYS and x.dtype == np.uint8 and x.ndim == 3:
                blobs.append(_image_bytes(x, jpeg))
                return {"__img__": len(blobs) - 1}
            a = np.ascontiguousarray(x)
            blobs.append(a.tobytes())
            return {"__nd__": len(blobs) - 1, "dtype": a.dtype.str, "shape": list(a.shape)}
        raise TypeError(f"cannot send {type(x).__name__}")

    header = json.dumps({"m": enc(msg), "n": [len(b) for b in blobs]}).encode()
    return b"".join([struct.pack("<I", len(header)), header] + blobs)


def decode(data: bytes):
    (n,) = struct.unpack_from("<I", data, 0)
    header = json.loads(data[4:4 + n].decode())
    blobs, off = [], 4 + n
    for size in header["n"]:
        blobs.append(data[off:off + size])
        off += size

    def dec(x):
        if isinstance(x, dict):
            if "__nd__" in x:
                return np.frombuffer(blobs[x["__nd__"]], dtype=np.dtype(x["dtype"])).reshape(x["shape"]).copy()
            if "__img__" in x:
                return _image(blobs[x["__img__"]])
            if "__f__" in x:
                return float(x["__f__"])
            return {k: dec(v) for k, v in x.items()}
        if isinstance(x, list):
            return [dec(v) for v in x]
        return x
    return dec(header["m"])
