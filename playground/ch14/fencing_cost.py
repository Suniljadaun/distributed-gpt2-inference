"""
fencing_cost.py - Chapter 14, Measure it
1. What do the safety checks cost per step (generation fencing + retry fingerprints)?
2. What does coalescing save when a client sends the same request several times at once?
Run:  python -m playground.ch14.fencing_cost            (about 1 minute)
"""
import statistics
import tempfile
import threading
import time
import timeit
from pathlib import Path

import grpc
import torch

from src.coordinator.cluster import supervised_cluster, two_stage_layout
from src.coordinator.server import start_coordinator
from src.coordinator.token_log import TokenLog
from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc
from src.proto.tensor_codec import tensor_to_proto
from src.worker.server import step_digest

PROMPT = [464, 3139, 286, 4881, 318]


def decode_ms(pipe, rid: str, steps: int = 60) -> float:
    tok, _ = pipe.step(rid, 0, PROMPT, 0)
    times = []
    for s in range(1, steps + 1):
        tok, timing = pipe.step(rid, s, [tok], len(PROMPT) + s - 1)
        times.append(timing.total_ms)
    pipe.release(rid)
    return statistics.median(times)


def burst(stub, ids: list[str], tokens: int) -> float:
    start = time.monotonic()
    threads = [threading.Thread(target=stub.Generate, args=(pb.GenerateRequest(
        request_id=rid, prompt_ids=PROMPT, max_new_tokens=tokens),), kwargs={"timeout": 120}) for rid in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.monotonic() - start


def main() -> None:
    print("1. Cost of the safety checks")
    for label, shape in [("1-token hidden state (3 KB)", (1, 1, 768)), ("20-token prompt (60 KB)", (1, 20, 768))]:
        req = pb.ForwardRequest(request_id="x", step_id=1, input=tensor_to_proto(torch.randn(shape)), start_pos=5)
        us = timeit.timeit(lambda: step_digest(req), number=2000) / 2000 * 1e6
        print(f"  retry fingerprint of a {label:<28} {us:6.1f} us")

    with tempfile.TemporaryDirectory() as d, supervised_cluster(two_stage_layout(base_port=50321)) as (pipe, sup):
        fenced = decode_ms(pipe, "fenced")
        saved = list(pipe.expected)
        pipe.expected = [0, 0]                                     # turn generation fencing off
        unfenced = decode_ms(pipe, "unfenced")
        pipe.expected = saved
        print(f"  decode step with fencing {fenced:5.1f} ms, without {unfenced:5.1f} ms "
              f"(difference {fenced - unfenced:+.1f} ms, i.e. noise)")

        print("\n2. Five copies of the same 30-token request vs five different requests")
        log = TokenLog(Path(d) / "log.jsonl")
        server, scheduler = start_coordinator(pipe, port=50323, queue_size=8, health=sup, log=log)
        channel = grpc.insecure_channel("127.0.0.1:50323")
        stub = pb_grpc.CoordinatorStub(channel)
        same = burst(stub, ["dup"] * 5, 30)
        after_same = stub.Stats(pb.StatsRequest(), timeout=5)
        different = burst(stub, [f"distinct-{k}" for k in range(5)], 30)
        after_diff = stub.Stats(pb.StatsRequest(), timeout=5)
        print(f"  5 x same id     : {same:5.2f} s, generated {after_same.completed} time(s), "
              f"coalesced {after_same.coalesced}")
        print(f"  5 different ids : {different:5.2f} s, generated {after_diff.completed - after_same.completed} times")
        channel.close()
        server.stop(grace=None)
        scheduler.stop()
        log.close()


if __name__ == "__main__":
    main()
