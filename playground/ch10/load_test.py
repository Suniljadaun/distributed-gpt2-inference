"""
load_test.py - Chapter 10, Measure it
Open-loop load at increasing arrival rates against the coordinator: throughput, latency,
rejections, and Little's Law (L = lambda * W) checked against sampled queue lengths.
Then the same overload against an (almost) unbounded queue.
Run:  python -m playground.ch10.load_test            (about 2 minutes)
"""
import argparse
import random
import statistics
import threading
import time
from concurrent import futures

import grpc

from src.coordinator.cluster import running_cluster, two_stage_layout
from src.coordinator.server import start_coordinator
from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc

PROMPT = [464, 3139, 286, 4881, 318]
TOKENS = 10                    # per request: ~0.3 s of service on the two-worker pipeline
DEADLINE_S = 3.0               # every client waits at most this long


def one_request(stub, rid: str) -> tuple[str, float]:
    start = time.monotonic()
    try:
        stub.Generate(pb.GenerateRequest(request_id=rid, prompt_ids=PROMPT, max_new_tokens=TOKENS),
                      timeout=DEADLINE_S)
        outcome = "ok"
    except grpc.RpcError as e:
        outcome = {grpc.StatusCode.RESOURCE_EXHAUSTED: "rejected",
                   grpc.StatusCode.DEADLINE_EXCEEDED: "expired"}.get(e.code(), e.code().name)
    return outcome, time.monotonic() - start


def run_load(stub, rate: float, seconds: float, tag: str) -> dict:
    """Poisson arrivals at `rate` per second for `seconds`; sample in-system count every 50 ms."""
    samples, stop = [], threading.Event()

    def sample() -> None:
        while not stop.is_set():
            s = stub.Stats(pb.StatsRequest(), timeout=2)
            samples.append(s.queued + s.running)
            time.sleep(0.05)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    calls, start, i = [], time.monotonic(), 0
    with futures.ThreadPoolExecutor(max_workers=128) as pool:
        while time.monotonic() - start < seconds:
            calls.append(pool.submit(one_request, stub, f"{tag}-{i}"))
            i += 1
            time.sleep(random.expovariate(rate))           # gaps of a Poisson process
        results = [c.result() for c in calls]
    elapsed = time.monotonic() - start                     # until the LAST request finished
    stop.set()
    sampler.join()

    by = {k: [t for o, t in results if o == k] for k in ("ok", "rejected", "expired")}
    admitted = by["ok"] + by["expired"]
    w = statistics.mean(admitted) if admitted else 0.0     # W: mean time in system of admitted requests
    ok_sorted = sorted(by["ok"])
    return {
        "rate": rate, "sent": len(results), "ok": len(by["ok"]), "rejected": len(by["rejected"]),
        "expired": len(by["expired"]), "goodput": len(by["ok"]) / elapsed,
        "p50": ok_sorted[len(ok_sorted) // 2] if ok_sorted else float("nan"),
        "p99": ok_sorted[int(len(ok_sorted) * 0.99)] if ok_sorted else float("nan"),
        "L_sampled": statistics.mean(samples), "L_little": len(admitted) / elapsed * w,
    }


def show(r: dict, label: str) -> None:
    print(f"{label:<12} {r['rate']:>4.1f} {r['sent']:>5} {r['ok']:>5} {r['rejected']:>5} {r['expired']:>5} "
          f"{r['goodput']:>8.2f} {r['p50']:>6.2f} {r['p99']:>6.2f} {r['L_sampled']:>8.2f} {r['L_little']:>8.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=15.0)
    args = parser.parse_args()
    random.seed(0)

    with running_cluster(two_stage_layout(base_port=50151)) as pipe:
        bounded, bsched = start_coordinator(pipe, port=50153, queue_size=8)
        unbounded, usched = start_coordinator(pipe, port=50154, queue_size=1000)
        b_ch, u_ch = grpc.insecure_channel("127.0.0.1:50153"), grpc.insecure_channel("127.0.0.1:50154")
        b_stub, u_stub = pb_grpc.CoordinatorStub(b_ch), pb_grpc.CoordinatorStub(u_ch)

        one_request(b_stub, "warm-up")
        service = statistics.median(one_request(b_stub, f"solo-{k}")[1] for k in range(5))
        print(f"service time of one request alone: {service * 1000:.0f} ms  ->  capacity ~ {1 / service:.1f} req/s\n")
        print(f"{'queue':<12} {'rate':>4} {'sent':>5} {'ok':>5} {'rej':>5} {'exp':>5} "
              f"{'goodput':>8} {'p50 s':>6} {'p99 s':>6} {'L sampled':>8} {'L = λW':>8}")
        for rate in [1.0, 2.0, 3.0, 4.0, 6.0]:
            show(run_load(b_stub, rate, args.seconds, f"b{rate}"), "bounded 8")
        show(run_load(u_stub, 6.0, args.seconds, "u6"), "unbounded")

        for ch in (b_ch, u_ch):
            ch.close()
        for server, sched in ((bounded, bsched), (unbounded, usched)):
            server.stop(grace=None)
            sched.stop()


if __name__ == "__main__":
    main()
