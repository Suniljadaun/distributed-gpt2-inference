"""
throughput.py - Chapter 11, Measure it
How much more work does the same pair of workers do with requests in flight (slots),
with micro-batching (max_batch), and with both? A burst of 24 requests, every output checked.
Run:  python -m playground.ch11.throughput            (about 2 minutes)
"""
import json
import statistics
import time
from concurrent import futures

import grpc

from src.coordinator.cluster import WorkerSpec, running_cluster, two_stage_layout
from src.coordinator.server import start_coordinator
from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc
from src.reference import REFERENCE_FILE

REFERENCE = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
ENTRIES = REFERENCE["entries"]
TOKENS = REFERENCE["max_new_tokens"]          # 20 per request
BURST = 24                                    # requests submitted at the same moment

CONFIGS = [                                   # (label, slots, max_batch)
    ("1 at a time (Ch 10)", 1, 0),
    ("2 slots", 2, 0),
    ("4 slots", 4, 0),
    ("batch 4", 1, 4),
    ("batch 8", 1, 8),
    ("batch 24", 1, 24),
    ("2 groups x batch 4", 2, 4),
    ("2 groups x batch 12", 2, 12),
]


def burst(stub) -> tuple[float, list[float], bool]:
    jobs = [(f"t{time.monotonic_ns()}-{i}", ENTRIES[i % len(ENTRIES)]) for i in range(BURST)]

    def one(rid, entry):
        t0 = time.monotonic()
        reply = stub.Generate(pb.GenerateRequest(request_id=rid, prompt_ids=entry["prompt_ids"],
                                                 max_new_tokens=TOKENS), timeout=120)
        return time.monotonic() - t0, list(reply.new_ids) == entry["new_ids"]

    start = time.monotonic()
    with futures.ThreadPoolExecutor(max_workers=BURST) as pool:
        results = [f.result() for f in [pool.submit(one, rid, e) for rid, e in jobs]]
    return time.monotonic() - start, [r[0] for r in results], all(r[1] for r in results)


def measure(layout: list[WorkerSpec], title: str, base_port: int) -> None:
    print(f"\n{title}")
    print(f"{'config':<22} {'tokens/s':>9} {'speed-up':>9} {'mean latency s':>15} {'all match':>10}")
    with running_cluster(layout) as pipe:
        baseline = None
        for k, (label, slots, max_batch) in enumerate(CONFIGS):
            server, scheduler = start_coordinator(pipe, base_port + k, queue_size=64, slots=slots, max_batch=max_batch)
            channel = grpc.insecure_channel(f"127.0.0.1:{base_port + k}")
            elapsed, latencies, ok = burst(pb_grpc.CoordinatorStub(channel))
            channel.close()
            server.stop(grace=None)
            scheduler.stop()
            rate = BURST * TOKENS / elapsed
            baseline = baseline or rate
            print(f"{label:<22} {rate:>9.1f} {rate / baseline:>8.2f}x {statistics.mean(latencies):>15.2f} {str(ok):>10}")


def main() -> None:
    measure(two_stage_layout(base_port=50201), "Workers with PyTorch's default threads (6 each on a 6-core CPU)", 50210)
    a, b = two_stage_layout(base_port=50221)
    three = [WorkerSpec(a.worker_id, a.port, a.first, a.last, embed=True, extra_args=("--threads", "3")),
             WorkerSpec(b.worker_id, b.port, b.first, b.last, head=True, extra_args=("--threads", "3"))]
    measure(three, "Workers limited to 3 threads each (--threads 3)", 50230)


if __name__ == "__main__":
    main()
