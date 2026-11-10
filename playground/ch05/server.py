"""
server.py - Chapter 5
A tiny TCP "worker": it keeps one piece of state (a counter) and answers requests,
one client connection at a time. It can be told to be slow, or to crash.
Run alone:  python -m playground.ch05.server --port 50555     (Ctrl+C to stop)
Normally client.py starts it for you.
"""
import argparse
import os
import socket
import time

from playground.ch05.framing import recv_msg, send_msg


def handle(header: dict, payload: bytes, state: dict) -> tuple[dict, bytes]:
    op = header["op"]
    if op == "echo":
        return {"id": header["id"], "ok": True}, payload
    if op == "add":
        state["counter"] += header["amount"]               # 1. do the work (state changes NOW)
        print(f"    [server] add -> counter = {state['counter']}", flush=True)
        time.sleep(header.get("delay", 0.0))                # 2. then be slow before replying
        return {"id": header["id"], "ok": True, "counter": state["counter"]}, b""
    if op == "get":
        return {"id": header["id"], "ok": True, "counter": state["counter"]}, b""
    if op == "crash":
        print("    [server] crashing now!", flush=True)
        os._exit(1)                                         # die instantly, like a power cut
    return {"id": header["id"], "ok": False, "error": f"unknown op {op!r}"}, b""


def serve(port: int) -> None:
    state = {"counter": 0}
    with socket.create_server(("127.0.0.1", port)) as listener:
        print(f"    [server pid={os.getpid()}] listening on 127.0.0.1:{port}", flush=True)
        while True:
            conn, _addr = listener.accept()                 # wait for the next client
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                while True:
                    try:
                        header, payload = recv_msg(conn)
                        send_msg(conn, *handle(header, payload, state))
                    except ConnectionError:                 # client closed or gave up: next client
                        break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50555)
    serve(parser.parse_args().port)


if __name__ == "__main__":
    main()
