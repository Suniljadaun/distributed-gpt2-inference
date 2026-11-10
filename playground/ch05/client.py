"""
client.py - Chapter 5
Starts server.py as a separate process, then runs five experiments against it:
latency, message size, a timeout + retry that does the work twice, a crash, and a dead server.
Run:  python -m playground.ch05.client
"""
import json
import os
import random
import socket
import statistics
import subprocess
import sys
import time

from playground.ch05.framing import recv_msg, send_msg

PORT = 50555
SERVER_CMD = [sys.executable, "-m", "playground.ch05.server", "--port", str(PORT)]


def connect(timeout: float) -> socket.socket:
    sock = socket.create_connection(("127.0.0.1", PORT), timeout=timeout)   # timeout covers every recv
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def call(sock: socket.socket, header: dict, payload: bytes = b"") -> tuple[dict, bytes]:
    """One request -> one reply. Raises TimeoutError or ConnectionError on trouble."""
    send_msg(sock, header, payload)
    return recv_msg(sock)


def start_server() -> subprocess.Popen:
    proc = subprocess.Popen(SERVER_CMD)
    deadline = time.monotonic() + 10
    while True:                                  # the server needs a moment to start listening
        try:
            connect(timeout=1.0).close()
            return proc
        except ConnectionRefusedError:
            if time.monotonic() > deadline or proc.poll() is not None:
                raise RuntimeError("server did not start")
            time.sleep(0.1)


def exp_latency() -> None:
    print("\n1. Round-trip time on localhost (1000 small requests)")
    with connect(timeout=2.0) as sock:
        times = []
        for i in range(1000):
            t0 = time.perf_counter()
            call(sock, {"id": i, "op": "echo"}, b"x" * 100)
            times.append(time.perf_counter() - t0)
    times.sort()
    print(f"  p50 {times[500] * 1e6:5.0f} us   p99 {times[990] * 1e6:5.0f} us   max {times[-1] * 1e6:6.0f} us")


def exp_sizes() -> None:
    print("\n2. Message size vs round-trip time (payload goes there AND back)")
    with connect(timeout=5.0) as sock:
        for label, n in [("100 B", 100), ("3 KB  = 1 token", 3072), ("60 KB = 20-token prompt", 61440),
                         ("1 MB", 1 << 20), ("10 MB", 10 << 20)]:
            payload = os.urandom(n)
            times = []
            for i in range(20):
                t0 = time.perf_counter()
                _, back = call(sock, {"id": i, "op": "echo"}, payload)
                times.append(time.perf_counter() - t0)
            assert back == payload                              # bytes arrive intact and in order
            t = statistics.median(times)
            print(f"  {label:<24} {t * 1e3:8.3f} ms   {2 * n / t / 1e9:6.2f} GB/s")
    vec = [random.gauss(0, 1) for _ in range(768)]
    print(f"  one hidden state as JSON text: {len(json.dumps(vec)):,} bytes   as raw float32: {768 * 4:,} bytes")


def exp_timeout_retry() -> None:
    print("\n3. Timeout + retry of a request that is NOT safe to repeat")
    request = {"id": 7, "op": "add", "amount": 1, "delay": 1.0}
    with connect(timeout=0.5) as sock:
        try:
            call(sock, request)
            print("  reply arrived")
        except TimeoutError:
            print("  no reply within 0.5 s. Did the add happen? The client cannot know. Retrying...")
    with connect(timeout=3.0) as sock:                          # retry with a longer timeout
        reply, _ = call(sock, request)
        print(f"  retry succeeded: {reply}")
        reply, _ = call(sock, {"id": 8, "op": "get"})
    print(f"  the client wanted +1 once; the counter is {reply['counter']}")


def exp_crash(proc: subprocess.Popen) -> None:
    print("\n4. The server crashes while handling a request")
    with connect(timeout=2.0) as sock:
        try:
            call(sock, {"id": 9, "op": "crash"})
        except ConnectionError as e:
            print(f"  client sees: {type(e).__name__}: {e}")
    proc.wait(timeout=5)
    print(f"  server process exit code: {proc.returncode}")

    print("\n5. Calling a server that is no longer there")
    t0 = time.perf_counter()
    try:
        connect(timeout=5.0)
    except ConnectionRefusedError as e:
        print(f"  client sees: {type(e).__name__} after {time.perf_counter() - t0:.2f} s: {e}")


def main() -> None:
    proc = start_server()
    try:
        exp_latency()
        exp_sizes()
        exp_timeout_retry()
        exp_crash(proc)
    finally:
        if proc.poll() is None:                                 # always clean up the child
            proc.terminate()
            proc.wait()


if __name__ == "__main__":
    main()
