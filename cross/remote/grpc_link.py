"""A real network between the edge and the server of a remote session (cross.remote): one bidirectional gRPC stream
per session, carrying the payloads of cross.remote.codec (method /cross.remote.Remote/Session, raw bytes both ways, no
generated stubs).

On the stream the edge sends {"op": "open", ...} (what the server needs to build the session), then one {"op":
"frame", ...} per frame (cross.remote.server.MapServer.handle), and control messages ("save_map", "load_map",
"keyframes", "close"); the server answers each in order.  GrpcLink is the edge's link (send / poll as the simulated
link, cross.remote.link.SimLink, but in wall-clock time); extra_delay adds a fixed one-way delay in each direction (a
wide-area network emulated on a local one).  serve() runs the server: build(open message) -> MapServer per stream."""

import queue
import threading
import time
from concurrent import futures

from .codec import decode, encode

METHOD = "/cross.remote.Remote/Session"
OPTIONS = [("grpc.max_send_message_length", 256 << 20), ("grpc.max_receive_message_length", 256 << 20)]


def serve(port: int, build, max_sessions: int = 4):
    """Serve sessions on 0.0.0.0:port until interrupted.  build(open_message) -> (MapServer, service_factory or None,
    reply dict for the open)."""
    import grpc

    def session(request_iterator, context):
        server = factory = None
        kf_frames, last_kf = {}, None
        max_backlog = None
        inbox = queue.Queue()                    # (time received, payload): a message's wait on this server

        def read():
            try:
                for raw in request_iterator:
                    inbox.put((time.monotonic(), raw))
            finally:
                inbox.put(None)
        threading.Thread(target=read, daemon=True).start()
        while True:
            item = inbox.get()
            if item is None:
                break
            received, raw = item
            msg = decode(raw)
            op = msg.pop("op")
            if op == "frame":
                stale = max_backlog is not None and time.monotonic() - received > max_backlog
                r = server.handle(msg, stale=stale) if stale else server.handle(msg)
                if r.get("last_added_kf_id") is not None and r["last_added_kf_id"] != last_kf:
                    last_kf = r["last_added_kf_id"]
                    kf_frames[int(last_kf)] = int(msg["index"])
                r.pop("timestamp", None)
                yield encode({"op": "reply", **r})
            elif op == "open":
                mb = msg.pop("max_backlog", None)
                max_backlog = None if mb is None or mb <= 0 else float(mb)
                server, factory, info = build(msg)
                yield encode({"op": "opened", **info})
            elif op == "save_map":
                server.save_map(msg["path"])
                yield encode({"op": "saved"})
            elif op == "load_map":
                if msg.get("new_service") and factory is not None:
                    server.service = factory()
                server.load_map(msg["path"])
                kf_frames, last_kf = {}, server.system.last_added_kf_id
                yield encode({"op": "loaded", "n_keyframes": len(server.system.hypothesis_manager.nodes)})
            elif op == "keyframes":
                from .server import pose_matrix
                nodes = server.system.hypothesis_manager.nodes
                yield encode({"op": "keyframes", "frames": {str(k): v for k, v in kf_frames.items()},
                              "nodes": {str(k): {"temporary": bool(kf.temporary), "pose": pose_matrix(kf.pose_mu[0])}
                                        for k, kf in nodes.items()}})
            elif op == "close":
                stats = dict(server.stats)
                server.release()
                yield encode({"op": "closed", "stats": stats})
                return
        if server is not None and server.system is not None:
            server.release()

    handler = grpc.method_handlers_generic_handler("cross.remote.Remote", {
        "Session": grpc.stream_stream_rpc_method_handler(session, request_deserializer=None, response_serializer=None)})
    srv = grpc.server(futures.ThreadPoolExecutor(max_workers=max_sessions), options=OPTIONS)
    srv.add_generic_rpc_handlers((handler,))
    srv.add_insecure_port(f"0.0.0.0:{port}")
    srv.start()
    return srv


class GrpcLink:
    """The edge's end of a session's stream.  send(message, t) / poll(t) as SimLink (t, the dataset time, is ignored:
    replies are returned when they have arrived); call(op, ...) for the control messages (blocks for the answer)."""

    def __init__(self, address: str, open_msg: dict, jpeg: int = 0, extra_delay: float = 0.0, timeout: float = 600.0,
                 realtime: bool = False):
        import grpc
        self.jpeg, self.extra_delay, self.timeout = int(jpeg or 0), float(extra_delay), float(timeout)
        self.realtime = bool(realtime)
        self._clock = None                       # (wall time, dataset time) of the first paced frame
        self.channel = grpc.insecure_channel(address, options=OPTIONS)
        grpc.channel_ready_future(self.channel).result(timeout=timeout)
        self._out = queue.Queue()
        self._frames = []                        # (release time, reply) of frame replies, in order
        self._control = queue.Queue()
        self._lock = threading.Lock()
        self._sent = {}                          # frame index -> wall time sent
        self._entries = {}
        self.log = []
        self.bytes_up = self.bytes_down = 0
        self.last_added_kf_id = None
        call = self.channel.stream_stream(METHOD, request_serializer=None, response_deserializer=None)
        self._responses = call(self._requests())
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self.opened = self.call("open", **open_msg)

    def _requests(self):
        while True:
            item = self._out.get()
            if item is None:
                return
            release, data = item
            wait = release - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            yield data

    def _read(self):
        try:
            for raw in self._responses:
                now = time.monotonic()
                self.bytes_down += len(raw)
                r = decode(raw)
                if r.get("op") == "reply":
                    with self._lock:
                        self._frames.append((now + 0.5 * self.extra_delay, r))
                else:
                    self._control.put(r)
        except Exception as e:                   # the stream ended (closed, or the server went away)
            self._control.put({"op": "error", "error": repr(e)})

    def pace(self, t):
        """Real time: wait until the frame of dataset time t is due (as a sensor would deliver it)."""
        if not self.realtime:
            return
        now = time.monotonic()
        if self._clock is None:
            self._clock = (now, t)
            return
        wait = self._clock[0] + (t - self._clock[1]) - now
        if wait > 0:
            time.sleep(wait)
        else:
            self.late = getattr(self, "late", 0) + int(wait < -0.05)

    def send(self, msg, t):
        data = encode({"op": "frame", **msg}, jpeg=self.jpeg)
        self.bytes_up += len(data)
        now = time.monotonic()
        self._sent[msg["index"]] = now
        entry = {"index": msg["index"], "sent": now, "bytes": len(data), "images": msg.get("rgb") is not None,
                 "request": msg.get("request") is not None}
        self.log.append(entry)
        self._entries[msg["index"]] = entry
        self._out.put((now + 0.5 * self.extra_delay, data))

    def poll(self, t=None):
        now = time.monotonic()
        out = []
        with self._lock:
            while self._frames and self._frames[0][0] <= now:
                r = self._frames.pop(0)[1]
                self._sent.pop(r["index"], None)
                entry = self._entries.pop(r["index"], None)
                if entry is not None:
                    entry["arrival"], entry["server_s"] = now, r.get("server_seconds")
                if r.get("last_added_kf_id") is not None:
                    self.last_added_kf_id = r["last_added_kf_id"]
                work = r.get("work") or {}
                self.n_stale = getattr(self, "n_stale", 0) + int(bool(work.get("stale")))
                self.n_observed = getattr(self, "n_observed", 0) + int(bool(work.get("observed")))
                out.append(r)
        return out

    def flush(self):
        """Wait until every frame sent has its reply (they stay queued for poll)."""
        t_end = time.monotonic() + self.timeout
        while self._sent and time.monotonic() < t_end:
            with self._lock:
                pending = {r["index"] for _, r in self._frames}
            if all(i in pending for i in self._sent):
                return
            time.sleep(0.005)

    def call(self, op, **kw):
        self._out.put((time.monotonic(), encode({"op": op, **kw})))
        r = self._control.get(timeout=self.timeout)
        if r.get("op") == "error":
            raise RuntimeError(f"remote session: {r['error']}")
        return r

    def reset(self):
        self._clock = None
        with self._lock:
            self._frames.clear()
            self._sent.clear()
            self._entries.clear()

    def close(self):
        try:
            out = self.call("close")
        finally:
            self._out.put(None)
            self.channel.close()
        return out

    def summary(self) -> dict:
        import numpy as np
        lat = np.array([e["arrival"] - e["sent"] for e in self.log if e.get("arrival") is not None])
        dur = max(self.log[-1]["sent"] - self.log[0]["sent"], 1e-9) if self.log else 1e-9
        out = {"extra_delay": self.extra_delay, "jpeg": self.jpeg, "messages": len(self.log), "realtime": self.realtime,
               "late_frames": getattr(self, "late", 0), "stale": getattr(self, "n_stale", 0),
               "observed": getattr(self, "n_observed", 0),
               "images": int(sum(e["images"] for e in self.log)), "uplink_kBps": self.bytes_up / dur / 1e3,
               "downlink_kBps": self.bytes_down / dur / 1e3}
        if len(lat):
            out["latency_s"] = {"mean": float(lat.mean()), "p50": float(np.median(lat)),
                                "p95": float(np.percentile(lat, 95)), "max": float(lat.max())}
        t0 = self.log[0]["sent"] if self.log else 0.0
        # per message: wall time sent, reply arrival - sent (None: none yet), server seconds, bytes up
        out["timeline"] = [[round(e["sent"] - t0, 3), None if e.get("arrival") is None else round(e["arrival"] - e["sent"], 4),
                            None if e.get("server_s") is None else round(e["server_s"], 4), int(e["bytes"])] for e in self.log]
        return out
