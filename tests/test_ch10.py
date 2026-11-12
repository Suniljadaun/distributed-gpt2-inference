"""
test_ch10.py - Chapter 10
Retries with backoff survive flaky and slow workers without changing the output; deadlines are
enforced and clean up; the coordinator's bounded queue rejects fast and never runs expired work.
Run:  python -m pytest tests/test_ch10.py -v
"""
import json
import math
import time
from concurrent import futures

import grpc
import pytest

from src.coordinator.cluster import WorkerSpec, running_cluster, two_stage_layout
from src.coordinator.pipeline import DeadlineExceeded, backoff_delay, generate
from src.coordinator.server import start_coordinator
from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc
from src.reference import REFERENCE_FILE

REFERENCE = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
ENTRY = REFERENCE["entries"][0]


def faulty_layout(base_port: int, *b_args: str) -> list[WorkerSpec]:
    a, b = two_stage_layout(base_port=base_port)
    return [a, WorkerSpec(b.worker_id, b.port, b.first, b.last, head=True, extra_args=b_args)]


# ---------------- backoff: no processes needed ----------------

def test_backoff_grows_exponentially_and_is_capped() -> None:
    for attempt in range(8):
        limit = min(1.0, 0.05 * 2 ** attempt)
        samples = [backoff_delay(attempt, 0.05, 1.0) for _ in range(200)]
        assert all(0.0 <= s <= limit for s in samples)
        assert max(samples) > 0.5 * limit                         # jitter really spreads over the range


# ---------------- retries and deadlines against faulty workers ----------------

def test_retries_hide_a_flaky_worker() -> None:
    """Worker B fails every 3rd call with UNAVAILABLE; the output must still match the reference."""
    with running_cluster(faulty_layout(50121, "--fail-every", "3")) as pipe:
        new_ids, timings = generate(pipe, "flaky", ENTRY["prompt_ids"], REFERENCE["max_new_tokens"])
    assert new_ids == ENTRY["new_ids"]
    assert sum(t.retries for t in timings) >= 5                   # ~1 in 3 of B's 20 calls failed


def test_late_replies_are_retried_from_the_stored_reply() -> None:
    """Every 4th reply from B arrives 600 ms late; calls time out at 300 ms and are retried."""
    with running_cluster(faulty_layout(50123, "--slow-every", "4", "--slow-ms", "600"), timeout_s=0.3) as pipe:
        new_ids, timings = generate(pipe, "late", ENTRY["prompt_ids"], REFERENCE["max_new_tokens"])
    assert new_ids == ENTRY["new_ids"]                            # no step was applied twice
    assert sum(t.retries for t in timings) >= 3


def test_request_deadline_is_enforced_and_caches_released() -> None:
    """Every reply from B is late, so 20 steps cannot fit in 1 s."""
    with running_cluster(faulty_layout(50125, "--slow-every", "1", "--slow-ms", "300"), timeout_s=0.2) as pipe:
        start = time.monotonic()
        with pytest.raises(DeadlineExceeded):
            generate(pipe, "too-slow", ENTRY["prompt_ids"], 20, deadline=start + 1.0)
        assert time.monotonic() - start < 1.5                     # gave up close to the deadline
        infos = [stub.Info(pb.InfoRequest(), timeout=5) for stub in pipe.stubs]
    assert [i.cached_requests for i in infos] == [0, 0]           # generate's finally released them


# ---------------- the coordinator's front door ----------------

@pytest.fixture(scope="module")
def front_door():
    with running_cluster(two_stage_layout(base_port=50141)) as pipe:
        server, scheduler = start_coordinator(pipe, port=50143, queue_size=1, slots=1)
        channel = grpc.insecure_channel("127.0.0.1:50143")
        yield pb_grpc.CoordinatorStub(channel)
        channel.close()
        server.stop(grace=None)
        scheduler.stop()


def call(stub, rid: str, max_new_tokens: int = 20, timeout: float = 10.0):
    return stub.Generate(pb.GenerateRequest(request_id=rid, prompt_ids=ENTRY["prompt_ids"],
                                            max_new_tokens=max_new_tokens), timeout=timeout)


def test_coordinator_generates_the_reference(front_door) -> None:
    reply = call(front_door, "door-1")
    assert list(reply.new_ids) == ENTRY["new_ids"]
    assert reply.service_ms > 0 and reply.queue_ms >= 0


def test_bad_request_is_invalid_argument(front_door) -> None:
    with pytest.raises(grpc.RpcError) as err:
        front_door.Generate(pb.GenerateRequest(request_id="empty", max_new_tokens=5), timeout=5)
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_full_queue_rejects_fast(front_door) -> None:
    """1 running + 1 queued fit; the other 4 are turned away at once."""
    with futures.ThreadPoolExecutor(max_workers=6) as pool:
        calls = [pool.submit(timed_call, front_door, f"burst-{i}") for i in range(6)]
        results = [c.result() for c in calls]
    ok = [r for r in results if r[0] == grpc.StatusCode.OK]
    rejected = [r for r in results if r[0] == grpc.StatusCode.RESOURCE_EXHAUSTED]
    assert len(ok) == 2 and len(rejected) == 4
    assert max(seconds for _, seconds in rejected) < 0.2          # rejection costs no waiting


def timed_call(stub, rid: str) -> tuple[grpc.StatusCode, float]:
    start = time.monotonic()
    try:
        call(stub, rid)
        return grpc.StatusCode.OK, time.monotonic() - start
    except grpc.RpcError as e:
        return e.code(), time.monotonic() - start


def test_request_expiring_in_the_queue_is_never_run(front_door) -> None:
    before = front_door.Stats(pb.StatsRequest(), timeout=5)
    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        long_one = pool.submit(call, front_door, "long", 20)      # occupies the only slot ~0.5 s
        time.sleep(0.05)
        short = pool.submit(timed_call_with_deadline, front_door, "impatient", 0.2)
        assert short.result()[0] == grpc.StatusCode.DEADLINE_EXCEEDED
        long_one.result()
    time.sleep(0.1)
    after = front_door.Stats(pb.StatsRequest(), timeout=5)
    assert after.expired == before.expired + 1                    # dropped from the queue ...
    assert after.completed == before.completed + 1                # ... only "long" was generated


def timed_call_with_deadline(stub, rid: str, timeout: float) -> tuple[grpc.StatusCode, float]:
    start = time.monotonic()
    try:
        call(stub, rid, 5, timeout=timeout)
        return grpc.StatusCode.OK, time.monotonic() - start
    except grpc.RpcError as e:
        return e.code(), time.monotonic() - start
