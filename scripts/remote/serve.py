#!/usr/bin/env python3
"""Server of remote CROSS sessions (cross/remote): the back end and the GPU work of each session an edge opens over gRPC.

  python scripts/remote/serve.py --port 50051            # on the GPU machine
  python scripts/map_and_reloc.py ... --remote-server HOST:50051 [--remote-realtime --remote-extra-delay 0.1]

The edge (the runner with --remote-server) sends the resolved session (mode, odometry, camera, configurations) when it
opens the stream; this process builds that session (cross.pipeline.server_session) and steps the edge's frames.  Map
files are read and written at the paths the edge names, on this machine.
"""
import argparse
import time

from loguru import logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--sessions", type=int, default=2, help="sessions served at once")
    a = ap.parse_args()
    from cross.pipeline import server_session
    from cross.remote.grpc_link import serve

    def build(msg):
        logger.info(f"opening a session: {msg['mode']} / {msg['odometry']}")
        return server_session(msg)
    srv = serve(a.port, build, max_sessions=a.sessions)
    logger.info(f"serving remote CROSS sessions on port {a.port}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.stop(5)


if __name__ == "__main__":
    main()
