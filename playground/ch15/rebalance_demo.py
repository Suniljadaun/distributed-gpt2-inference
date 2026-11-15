"""
rebalance_demo.py - Chapter 15, Measure it
Throughput over time under steady load, with and without dynamic rebalancing.
Each "machine" gets its own CPU cores: worker A cores 0-1, worker B cores 2-3, this coordinator 4-5.
Workers compute one step at a time (--serial): a pipeline stage is a single server.
Both start with layers 0-2 | 3-5. Some seconds in, worker A gets slower (+8 ms per block, a noisy
neighbour). With the rebalancer, layers move to whoever is faster; without it, the split stays.
policy: None = static split, "model" = predict the bottleneck (the engine's), "threshold" = the scope report's
imbalance-ratio rule, for comparison.
Run:  python -m playground.ch15.rebalance_demo            (about 2.5 minutes)
"""
import statistics
import threading
import time

import grpc

from src.coordinator.cluster import WorkerSpec, running_cluster, two_stage_layout
from src.coordinator.rebalancer import Rebalancer
from src.coordinator.server import start_coordinator
from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc
from src.worker.server import pin_to_cpus

PROMPT = [464, 3139, 286, 4881, 318]
RUN_S, SLOW_AT_S, BUCKET_S = 48, 24, 4
SLOTS, CLIENTS = 4, 8              # enough requests in flight to keep both stages busy
DELAY_MS = 8                       # how much slower each block on A becomes


def run(policy: str | None, base_port: int) -> tuple[list[tuple[int, float, str]], list[tuple[float, str]]]:
    """Returns the timeline (t, tokens/s, detail) and the rebalancer's moves (t, text)."""
    a, b = two_stage_layout(base_port=base_port)
    common = ("--threads", "2", "--serial")
    layout = [WorkerSpec(a.worker_id, a.port, a.first, a.last, embed=True,
                         extra_args=common + ("--cpus", "0,1", "--block-delay-ms", str(DELAY_MS),
                                              "--block-delay-after-s", str(SLOW_AT_S + 6))),   # +6 s: start-up
              WorkerSpec(b.worker_id, b.port, b.first, b.last, head=True, extra_args=common + ("--cpus", "2,3"))]
    rows = []
    with running_cluster(layout) as pipe:
        reb = Rebalancer(pipe, period_s=1.0, window_s=3.0, cooldown_s=3.0, min_samples=15,
                         policy=policy or "model")
        server, scheduler = start_coordinator(pipe, base_port + 2, queue_size=64, slots=SLOTS)
        channel = grpc.insecure_channel(f"127.0.0.1:{base_port + 2}")
        stub = pb_grpc.CoordinatorStub(channel)
        t0 = time.monotonic()
        stop = t0 + RUN_S

        def client(c: int) -> None:
            k = 0
            while time.monotonic() < stop:
                stub.Generate(pb.GenerateRequest(request_id=f"c{c}-{k}", prompt_ids=PROMPT, max_new_tokens=30),
                              timeout=60)
                k += 1

        threads = [threading.Thread(target=client, args=(c,)) for c in range(CLIENTS)]
        for t in threads:
            t.start()
        if policy is not None:
            reb.start()
        for bucket in range(RUN_S // BUCKET_S):
            time.sleep(max(0.0, t0 + (bucket + 1) * BUCKET_S - time.monotonic()))
            since = t0 + bucket * BUCKET_S
            steps = pipe.recent(since)
            tokens = sum(s[1] for s in steps)
            a_ms = statistics.median(s[2][0] for s in steps) if steps else 0
            b_ms = statistics.median(s[2][1] for s in steps) if steps else 0
            layout_now = " | ".join(f"{f}-{l - 1}" for f, l in reb.layout)
            rows.append((int(since - t0), tokens / BUCKET_S, f"A {a_ms:4.1f} B {b_ms:4.1f} ms  {layout_now}"))
        for t in threads:
            t.join()
        reb.stop()
        channel.close()
        server.stop(grace=None)
        scheduler.stop()
    moves = [(round(t - t0, 1), text) for t, text in reb.events if text.startswith("moved layer")]
    return rows, moves


def main() -> None:
    pin_to_cpus([4, 5])                                        # the coordinator and its clients
    results = {}
    for name, policy, port in [("static split", None, 50341), ("dynamic rebalancing", "model", 50345)]:
        print(f"\n=== {name} ===", flush=True)
        results[name], _ = run(policy, port)
    print(f"\nA becomes slower (+{DELAY_MS} ms per block) at about t = {SLOW_AT_S} s\n")
    print(f"{'t (s)':>6} | {'static tok/s':>12}  {'':<30} | {'dynamic tok/s':>13}  {''}")
    for (t, s_rate, s_info), (_, d_rate, d_info) in zip(results["static split"], results["dynamic rebalancing"]):
        print(f"{t:>6} | {s_rate:>12.1f}  {s_info:<30} | {d_rate:>13.1f}  {d_info}")
    for name in results:
        before = [r for t, r, _ in results[name] if 8 <= t < SLOW_AT_S]
        after = [r for t, r, _ in results[name] if t >= SLOW_AT_S + 8]
        print(f"{name:<20}: mean {statistics.mean(before):6.1f} tok/s before the slowdown, "
              f"{statistics.mean(after):6.1f} tok/s after it settles")


if __name__ == "__main__":
    main()
