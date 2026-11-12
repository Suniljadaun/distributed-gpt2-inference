"""
test_ch11.py - Chapter 11
Batched decode gives the same tokens as one-at-a-time; several requests in flight (slots) and
continuous batching through the coordinator still reproduce the reference for every request.
Run:  python -m pytest tests/test_ch11.py -v
"""
import json
from concurrent import futures

import grpc
import pytest
import torch

from src.coordinator.cluster import running_cluster, two_stage_layout
from src.coordinator.server import start_coordinator
from src.model.kv_cache import KVCache
from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc
from src.reference import REFERENCE_FILE
from src.worker.stage import load_stage

REFERENCE = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
ENTRIES = REFERENCE["entries"]
N = REFERENCE["max_new_tokens"]


# ---------------- in one process ----------------

@torch.inference_mode()
def test_batched_decode_matches_reference_in_process() -> None:
    a, b = load_stage(0, 3, True, False), load_stage(3, 6, False, True)
    caches_a, caches_b = [KVCache() for _ in ENTRIES], [KVCache() for _ in ENTRIES]
    tokens, positions = [], []
    for e, ca, cb in zip(ENTRIES, caches_a, caches_b):                  # prefill one by one
        tokens.append(int(b(a(torch.tensor([e["prompt_ids"]]), 0, ca), 0, cb)[0, -1].argmax()))
        positions.append(len(e["prompt_ids"]))
    outputs = [[t] for t in tokens]
    for _ in range(N - 1):                                              # then decode all 3 as ONE batch
        logits = b.forward_decode_batch(a.forward_decode_batch(torch.tensor([[t] for t in tokens]),
                                                               positions, caches_a), positions, caches_b)
        tokens = [int(t) for t in logits[:, -1].argmax(-1)]
        positions = [p + 1 for p in positions]
        for out, t in zip(outputs, tokens):
            out.append(t)
    assert outputs == [e["new_ids"] for e in ENTRIES]


# ---------------- through workers and the coordinator ----------------

@pytest.fixture(scope="module")
def pipe():
    with running_cluster(two_stage_layout(base_port=50161)) as p:
        yield p


def test_step_batch_across_workers(pipe) -> None:
    rids = [f"sb-{i}" for i in range(len(ENTRIES))]
    toks = [pipe.step(rid, 0, e["prompt_ids"], 0)[0] for rid, e in zip(rids, ENTRIES)]
    positions = [len(e["prompt_ids"]) for e in ENTRIES]
    outputs = [[t] for t in toks]
    for step in range(1, N):
        toks, _ = pipe.step_batch([(rid, step, t, p) for rid, t, p in zip(rids, toks, positions)])
        positions = [p + 1 for p in positions]
        for out, t in zip(outputs, toks):
            out.append(t)
    for rid in rids:
        pipe.release(rid)
    assert outputs == [e["new_ids"] for e in ENTRIES]


def test_retried_batch_returns_stored_replies(pipe) -> None:
    """Sending the same batch twice (a lost reply) must not advance any cache twice."""
    rids = ["rb-0", "rb-1"]
    toks = [pipe.step(rid, 0, ENTRIES[i]["prompt_ids"], 0)[0] for i, rid in enumerate(rids)]
    items = [(rid, 1, t, len(ENTRIES[i]["prompt_ids"])) for i, (rid, t) in enumerate(zip(rids, toks))]
    first, _ = pipe.step_batch(items)
    again, _ = pipe.step_batch(items)
    assert again == first == [ENTRIES[0]["new_ids"][1], ENTRIES[1]["new_ids"][1]]
    for rid in rids:
        pipe.release(rid)


def run_through_coordinator(pipe, port: int, slots: int, max_batch: int, copies: int = 3) -> list[list[int]]:
    """Send every reference prompt `copies` times, all at once; return outputs in submission order."""
    server, scheduler = start_coordinator(pipe, port, queue_size=32, slots=slots, max_batch=max_batch)
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = pb_grpc.CoordinatorStub(channel)
    jobs = [(f"c{port}-{k}-{i}", e) for k in range(copies) for i, e in enumerate(ENTRIES)]
    try:
        with futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            calls = [pool.submit(stub.Generate, pb.GenerateRequest(
                request_id=rid, prompt_ids=e["prompt_ids"], max_new_tokens=N), timeout=60) for rid, e in jobs]
            return [list(c.result().new_ids) for c in calls]
    finally:
        channel.close()
        server.stop(grace=None)
        scheduler.stop()


@pytest.mark.parametrize("slots, max_batch", [(2, 0), (1, 4), (2, 4)],
                         ids=["2-slots", "batch-4", "2-groups-of-4"])
def test_concurrent_requests_all_match_reference(pipe, slots, max_batch) -> None:
    port = 50170 + 2 * slots + max_batch
    outputs = run_through_coordinator(pipe, port, slots, max_batch)
    expected = [e["new_ids"] for _ in range(3) for e in ENTRIES]
    assert outputs == expected
    infos = [stub.Info(pb.InfoRequest(), timeout=5) for stub in pipe.stubs]
    assert all(i.cached_requests == 0 for i in infos)                 # every cache was released
