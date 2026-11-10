"""
client.py - Chapter 6
Starts the gRPC server as a separate process and repeats Chapter 5's experiments with gRPC:
latency, sizes (and the 4 MB limit), a deadline + safe retry, a bad argument,
concurrent calls, a streamed reply, and a dead server.
Run:  python -m playground.ch06.client
"""
import statistics
import subprocess
import sys
import time
from concurrent import futures

import grpc

from playground.ch06 import counter_pb2 as pb
from playground.ch06 import counter_pb2_grpc as pb_grpc

PORT = 50556
SERVER_CMD = [sys.executable, "-m", "playground.ch06.server", "--port", str(PORT)]


def start_server() -> tuple[subprocess.Popen, grpc.Channel, pb_grpc.CounterStub]:
    proc = subprocess.Popen(SERVER_CMD)
    channel = grpc.insecure_channel(f"127.0.0.1:{PORT}")
    grpc.channel_ready_future(channel).result(timeout=10)      # wait until the server is reachable
    return proc, channel, pb_grpc.CounterStub(channel)


def exp_latency(stub: pb_grpc.CounterStub) -> None:
    print("\n1. Round-trip time of a unary RPC (1000 small calls)")
    times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        stub.Echo(pb.EchoRequest(payload=b"x" * 100), timeout=2.0)
        times.append(time.perf_counter() - t0)
    times.sort()
    print(f"  p50 {times[500] * 1e6:5.0f} us   p99 {times[990] * 1e6:5.0f} us   max {times[-1] * 1e6:6.0f} us")


def exp_sizes(stub: pb_grpc.CounterStub) -> None:
    print("\n2. Message size vs round-trip time")
    for label, n in [("3 KB  = 1 token", 3072), ("60 KB = 20-token prompt", 61440),
                     ("1 MB", 1 << 20), ("10 MB", 10 << 20)]:
        payload = bytes(n)
        try:
            times = []
            for _ in range(20):
                t0 = time.perf_counter()
                stub.Echo(pb.EchoRequest(payload=payload), timeout=5.0)
                times.append(time.perf_counter() - t0)
            t = statistics.median(times)
            print(f"  {label:<24} {t * 1e3:8.3f} ms   {2 * n / t / 1e9:6.2f} GB/s")
        except grpc.RpcError as e:
            print(f"  {label:<24} FAILED  {e.code().name}: {e.details()}")


def exp_deadline_and_retry(stub: pb_grpc.CounterStub) -> None:
    print("\n3. Deadline exceeded, then a retry with the SAME request_id")
    request = pb.AddRequest(request_id="req-7", amount=1, delay_s=1.0)
    try:
        stub.Add(request, timeout=0.5)
    except grpc.RpcError as e:
        print(f"  first try: {e.code().name}  ({e.details()})")
    reply = stub.Add(request, timeout=3.0)
    print(f"  retry    : counter = {reply.counter}")
    time.sleep(0.7)                                            # let the first call's thread finish
    print(f"  Get      : counter = {stub.Get(pb.GetRequest(), timeout=1.0).counter}  (added once)")


def exp_bad_argument(stub: pb_grpc.CounterStub) -> None:
    print("\n4. A request the server rejects")
    try:
        stub.Add(pb.AddRequest(request_id="req-8", amount=-5), timeout=1.0)
    except grpc.RpcError as e:
        print(f"  {e.code().name}: {e.details()}")


def exp_concurrent(stub: pb_grpc.CounterStub) -> None:
    print("\n5. Four 1-second Adds at the same time")
    t0 = time.perf_counter()
    with futures.ThreadPoolExecutor(max_workers=4) as pool:
        calls = [pool.submit(stub.Add, pb.AddRequest(request_id=f"par-{i}", amount=10, delay_s=1.0), timeout=3.0)
                 for i in range(4)]
        results = sorted(c.result().counter for c in calls)
    print(f"  took {time.perf_counter() - t0:.2f} s   replies {results}")


def exp_stream(stub: pb_grpc.CounterStub) -> None:
    print("\n6. Server streaming: one request, replies arrive one by one")
    t0 = time.perf_counter()
    for tick in stub.CountUp(pb.CountUpRequest(n=5, interval_s=0.1), timeout=5.0):
        print(f"  tick {tick.i} at {(time.perf_counter() - t0) * 1e3:5.0f} ms")


def exp_dead_server(proc: subprocess.Popen, stub: pb_grpc.CounterStub) -> None:
    print("\n7. The server process is killed; then we call it")
    proc.kill()
    proc.wait()
    for attempt in (1, 2):                    # 1: on the old connection, 2: gRPC tries to reconnect
        t0 = time.perf_counter()
        try:
            stub.Get(pb.GetRequest(), timeout=5.0)
        except grpc.RpcError as e:
            print(f"  call {attempt}: {e.code().name} after {time.perf_counter() - t0:.2f} s: {e.details()[:60]}")


def main() -> None:
    proc, channel, stub = start_server()
    try:
        exp_latency(stub)
        exp_sizes(stub)
        exp_deadline_and_retry(stub)
        exp_bad_argument(stub)
        exp_concurrent(stub)
        exp_stream(stub)
        exp_dead_server(proc, stub)
    finally:
        channel.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    main()
