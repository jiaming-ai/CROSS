"""A simulated network between the edge and the server of a remote session (cross.remote), on the dataset's clock.

A message sent at frame time t reaches the server after half the round-trip time (plus jitter and its upload time at
the uplink bandwidth), waits until the server has finished the messages before it, takes the server's compute time, and
its reply reaches the edge half a round trip (plus jitter) later; replies arrive in order (one stream).  While the link
is down (an outage), messages and replies wait for it to come back.  The server runs in-process: it steps every message
as soon as it is sent (its results do not depend on time), and only the reply's arrival is simulated, so a run is
deterministic for a compute model and the same in every repetition.

Compute time: "model" (seconds per frame, observation, own pass, learned depth, stereo matching; COMPUTE_MODEL), the
"measured" wall time of the in-process server (this machine's GPU), or "zero"."""

import heapq

import numpy as np

# server seconds per message: a frame, plus an observation of the back end, an own pass of the odometry, learned depth,
# stereo matching (placeholders until measured; see outputs/2026-10-06_remote_mode)
COMPUTE_MODEL = {"frame": 0.004, "observe": 0.110, "own_pass": 0.040, "depth": 0.030, "stereo": 0.020}


def _jpeg(img, quality):
    import cv2
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].copy(), int(buf.size)


def image_bytes(img, quality=0):
    """Encoded size of an image: JPEG at the quality, else lossless PNG (an estimate of what a link would carry)."""
    import cv2
    if img is None:
        return 0
    if quality:
        return _jpeg(img, quality)[1]
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(img[..., ::-1]), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return int(buf.size)


class SimLink:
    def __init__(self, server, rtt=0.0, jitter=0.0, compute="model", costs=None, outages=(), jpeg=0,
                 uplink_mbps=None, seed=0, measure_bytes=True):
        self.server = server
        self.rtt, self.jitter = float(rtt), float(jitter)
        self.compute = compute
        self.costs = dict(COMPUTE_MODEL, **(costs or {}))
        self.outages = sorted((float(a), float(a) + float(d)) for a, d in outages)
        self.jpeg = int(jpeg or 0)
        self.uplink = None if not uplink_mbps else float(uplink_mbps) * 1e6 / 8.0       # bytes per second
        self.rng = np.random.default_rng(seed)
        self.measure_bytes = measure_bytes or self.uplink is not None
        self.t0 = None                           # dataset time of the first message (outages are relative to it)
        self.server_free = -np.inf
        self.last_arrival = -np.inf
        self._queue = []                         # (arrival, seq, reply)
        self._seq = 0
        self.log = []                            # per message: index, sent, at server, done, arrival, bytes, compute

    def _up(self, t):
        """The time a transmission started at t gets through (held during an outage)."""
        t_rel = t - self.t0
        for a, b in self.outages:
            if a <= t_rel < b:
                return self.t0 + b
        return t

    def _delay(self):
        half = 0.5 * self.rtt
        if self.jitter > 0:
            half += float(self.rng.exponential(self.jitter))
        return half

    def _cost(self, reply):
        if self.compute == "zero":
            return 0.0
        if self.compute == "measured":
            return float(reply.get("server_seconds", 0.0))
        w, c = reply.get("work", {}), self.costs
        return (c["frame"] + c["observe"] * w.get("observed", False) + c["own_pass"] * w.get("own_pass", False)
                + c["depth"] * w.get("depth", False) + c["stereo"] * w.get("stereo", False))

    def send(self, msg, t):
        if self.t0 is None:
            self.t0 = t
        size = 300
        if self.jpeg:
            for key in ("rgb", "rgb_right"):
                if msg.get(key) is not None:
                    msg[key], n = _jpeg(msg[key], self.jpeg)
                    size += n
        elif self.measure_bytes:
            for key in ("rgb", "rgb_right"):
                size += image_bytes(msg.get(key))
        if msg.get("depth") is not None:
            size += int(np.asarray(msg["depth"]).nbytes // 2)
        t_up = self._up(t) + self._delay() + (size / self.uplink if self.uplink else 0.0)
        reply = self.server.handle(msg)
        start = max(self._up(t_up), self.server_free)
        cost = self._cost(reply)
        self.server_free = start + cost
        arrival = self._up(self.server_free) + self._delay()
        arrival = max(arrival, self.last_arrival)           # one ordered stream
        self.last_arrival = arrival
        if self.rtt == 0 and self.jitter == 0 and self.compute == "zero" and not self.outages and not self.uplink:
            arrival = t                                     # the local session's timing
        heapq.heappush(self._queue, (arrival, self._seq, reply))
        self._seq += 1
        self.log.append({"index": msg["index"], "sent": t, "start": start, "done": self.server_free, "arrival": arrival,
                         "bytes": size, "cost": cost, "images": msg.get("rgb") is not None,
                         "request": msg.get("request") is not None, "measured": float(reply.get("server_seconds", 0.0)),
                         "work": dict(reply.get("work", {}))})

    def poll(self, t):
        out = []
        while self._queue and self._queue[0][0] <= t + 1e-9:
            out.append(heapq.heappop(self._queue)[2])
        return out

    def flush(self):
        pass

    def reset(self):
        """A new session (a map loaded for a relocalization trial): its clock starts again, nothing is in flight."""
        self.t0 = None
        self.server_free = self.last_arrival = -np.inf
        self._queue = []

    def summary(self) -> dict:
        if not self.log:
            return {}
        lat = np.array([e["arrival"] - e["sent"] for e in self.log])
        cost = np.array([e["cost"] for e in self.log])
        dur = max(self.log[-1]["sent"] - self.log[0]["sent"], 1e-9)
        up = np.array([e["bytes"] for e in self.log])
        # the measured server time of each kind of message (this machine): the compute model's evidence
        kinds = {}
        for e in self.log:
            w = e.get("work", {})
            k = "+".join(n for n in ("observed", "own_pass", "depth", "stereo") if w.get(n)) or "frame"
            kinds.setdefault(k, []).append(e["measured"])
        measured = {k: {"n": len(v), "mean": float(np.mean(v)), "p50": float(np.median(v)),
                        "p95": float(np.percentile(v, 95))} for k, v in kinds.items()}
        return {"rtt": self.rtt, "jitter": self.jitter, "compute": self.compute, "jpeg": self.jpeg,
                "measured_server_s": measured,
                "outages": self.outages, "uplink_mbps": None if not self.uplink else self.uplink * 8 / 1e6,
                "messages": len(self.log), "images": int(sum(e["images"] for e in self.log)),
                "requests": int(sum(e["request"] for e in self.log)),
                "latency_s": {"mean": float(lat.mean()), "p50": float(np.median(lat)), "p95": float(np.percentile(lat, 95)),
                              "max": float(lat.max())},
                "server_busy": float(cost.sum() / dur), "uplink_kBps": float(up.sum() / dur / 1e3),
                # per message: dataset time sent, reply arrival - sent, modelled and measured server seconds, bytes
                "timeline": [[round(e["sent"] - self.log[0]["sent"], 3), round(e["arrival"] - e["sent"], 4),
                              round(e["cost"], 4), round(e["measured"], 4), int(e["bytes"])] for e in self.log]}
