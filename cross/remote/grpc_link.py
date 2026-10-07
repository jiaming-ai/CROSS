"""A real network between the edge and the server of a remote session (cross.remote): one bidirectional gRPC stream
per session, carrying the payloads of cross_edge.codec (method /cross.remote.Remote/Session, raw bytes both ways, no
generated stubs).

On the stream the edge sends {"op": "open", ...} (what the server needs to build the session), then one {"op":
"frame", ...} per frame (cross.remote.server.MapServer.handle), and control messages ("save_map", "load_map",
"keyframes", "close"); the server answers each in order.  GrpcLink (cross_edge.grpc_client) is the edge's link (send / poll as the simulated
link, cross.remote.link.SimLink, but in wall-clock time); extra_delay adds a fixed one-way delay in each direction (a
wide-area network emulated on a local one).  serve() runs the server: build(open message) -> MapServer per stream."""

import queue
import threading
import time
from concurrent import futures

from cross_edge.codec import decode, encode
from cross_edge.grpc_client import METHOD, OPTIONS, GrpcLink  # noqa: F401  (GrpcLink: the edge's end, re-exported)


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
                yield encode({"op": "loaded", "n_keyframes": len(server.system.hypothesis_manager.nodes),
                              "cadence": server.cadence_state()})
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
