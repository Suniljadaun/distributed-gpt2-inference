"""
test_ch14.py - Chapter 14
Stale and conflicting messages are rejected: calls addressed to another worker generation are
fenced, a "retry" with different input is refused, duplicate requests are coalesced into one
computation, and a request id cannot be reused for a different request.
Run:  python -m pytest tests/test_ch14.py -v
"""
import json
import threading
import time

import grpc
import pytest
import torch

from src.coordinator.cluster import running_cluster, supervised_cluster, two_stage_layout
from src.coordinator.pipeline import StaleReply
from src.coordinator.server import start_coordinator
from src.coordinator.token_log import TokenLog
from src.model.generate import generate_cached
from src.model.weights import load_pretrained
from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc
from src.proto.tensor_codec import tensor_to_proto
from src.reference import REFERENCE_FILE

REFERENCE = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
ENTRY = REFERENCE["entries"][0]
PROMPT, N = ENTRY["prompt_ids"], REFERENCE["max_new_tokens"]


def ids_msg(ids: list[int]) -> pb.Tensor:
    return tensor_to_proto(torch.tensor([ids]))


@pytest.fixture(scope="module")
def pipe():
    with running_cluster(two_stage_layout(base_port=50311)) as p:
        yield p


# ---------------- fencing by generation ----------------

def test_call_for_another_generation_is_fenced(pipe) -> None:
    stale = pb.ForwardRequest(request_id="zombie-call", step_id=0, input=ids_msg(PROMPT), expect_generation=7)
    with pytest.raises(grpc.RpcError) as err:
        pipe.stubs[0].Forward(stale, timeout=5)                  # worker A is generation 1
    assert err.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert "fenced" in err.value.details()
    info = pipe.stubs[0].Info(pb.InfoRequest(), timeout=5)
    assert info.cached_requests == 0                             # fenced BEFORE any state was created


def test_pipeline_fences_its_calls_and_follows_the_right_generation(pipe) -> None:
    pipe.set_expected(1, 2)                                      # we wrongly believe B is generation 2
    try:
        with pytest.raises(grpc.RpcError) as err:
            pipe.step("believe-2", 0, PROMPT, 0)
        assert "fenced" in err.value.details()
    finally:
        pipe.set_expected(1, 1)                                  # B really is generation 1
        pipe.release("believe-2")
    tok, _ = pipe.step("believe-1", 0, PROMPT, 0)
    pipe.release("believe-1")
    assert tok == ENTRY["new_ids"][0]


def test_reply_from_an_unexpected_generation_is_refused(pipe, monkeypatch) -> None:
    """Defence in depth: even if a worker failed to fence, the coordinator checks the reply's generation."""
    real_forward = pipe._forward

    def old_worker(stub, request, deadline):                     # a worker that ignores the fencing field
        request.expect_generation = 0
        reply, r = real_forward(stub, request, deadline)
        return reply, r

    monkeypatch.setattr(pipe, "_forward", old_worker)
    pipe.set_expected(0, 3)
    try:
        with pytest.raises(StaleReply):
            pipe.step("old-worker", 0, PROMPT, 0)
    finally:
        pipe.set_expected(0, 0)
        pipe.release("old-worker")


# ---------------- idempotency must check the content ----------------

def test_conflicting_retry_is_rejected_but_a_true_retry_is_answered(pipe) -> None:
    tok0, _ = pipe.step("conflict", 0, PROMPT, 0)
    tok1, _ = pipe.step("conflict", 1, [tok0], len(PROMPT))
    with pytest.raises(grpc.RpcError) as err:
        pipe.step("conflict", 1, [tok0 + 1], len(PROMPT))       # step 1 again, but a different token
    assert err.value.code() == grpc.StatusCode.FAILED_PRECONDITION and "conflicting retry" in err.value.details()
    again, _ = pipe.step("conflict", 1, [tok0], len(PROMPT))     # the genuine retry still works
    assert again == tok1
    pipe.release("conflict")


# ---------------- duplicates at the front door ----------------

@pytest.fixture(scope="module")
def expected_60() -> list[int]:
    return generate_cached(load_pretrained(), PROMPT, 60)


def test_duplicate_requests_are_coalesced_and_ids_cannot_be_reused(tmp_path, expected_60) -> None:
    with supervised_cluster(two_stage_layout(base_port=50313)) as (p, sup):
        server, scheduler = start_coordinator(p, port=50315, queue_size=4, health=sup,
                                              log=TokenLog(tmp_path / "log.jsonl"))
        channel = grpc.insecure_channel("127.0.0.1:50315")
        stub = pb_grpc.CoordinatorStub(channel)
        request = pb.GenerateRequest(request_id="twice", prompt_ids=PROMPT, max_new_tokens=60)
        try:
            replies = {}

            def send(k: int) -> None:
                replies[k] = stub.Generate(request, timeout=60)

            threads = [threading.Thread(target=send, args=(k,)) for k in range(3)]
            for t in threads:
                t.start()
                time.sleep(0.1)                                  # the copies arrive while the first runs
            for t in threads:
                t.join()
            assert all(list(r.new_ids) == expected_60 for r in replies.values())
            stats = stub.Stats(pb.StatsRequest(), timeout=5)
            assert stats.admitted == 1 and stats.coalesced == 2 and stats.completed == 1   # computed once

            other = pb.GenerateRequest(request_id="twice", prompt_ids=[1, 2, 3], max_new_tokens=60)
            with pytest.raises(grpc.RpcError) as err:
                stub.Generate(other, timeout=5)
            assert err.value.code() == grpc.StatusCode.ALREADY_EXISTS
        finally:
            channel.close()
            server.stop(grace=None)
            scheduler.stop()


def test_worker_crash_with_fencing_still_gives_identical_output(tmp_path, expected_60) -> None:
    with supervised_cluster(two_stage_layout(base_port=50317)) as (p, sup):
        assert p.expected == [1, 1]
        server, scheduler = start_coordinator(p, port=50319, queue_size=4, health=sup,
                                              log=TokenLog(tmp_path / "log.jsonl"))
        channel = grpc.insecure_channel("127.0.0.1:50319")
        stub = pb_grpc.CoordinatorStub(channel)
        try:
            out = {}
            t = threading.Thread(target=lambda: out.setdefault("r", stub.Generate(pb.GenerateRequest(
                request_id="fenced-survivor", prompt_ids=PROMPT, max_new_tokens=60), timeout=60)))
            t.start()
            time.sleep(0.8)
            sup.kill(1)
            t.join(timeout=60)
            assert list(out["r"].new_ids) == expected_60 and out["r"].replays >= 1
            assert p.expected == [1, 2]                           # the pipeline now fences to B's new generation
        finally:
            channel.close()
            server.stop(grace=None)
            scheduler.stop()
