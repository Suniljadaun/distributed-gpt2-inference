"""
test_ch12.py - Chapter 12
The supervisor detects a crashed worker (process exit) and a hung one (missed heartbeats),
restarts it as a new generation, and the system serves correct output again. A request that
was in flight on the dead worker fails cleanly as UNAVAILABLE and leaves no cache behind.
Run:  python -m pytest tests/test_ch12.py -v
"""
import json
import threading
import time

import grpc
import pytest

from src.coordinator.cluster import WorkerSpec, supervised_cluster, two_stage_layout
from src.coordinator.pipeline import generate
from src.coordinator.server import start_coordinator
from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc
from src.reference import REFERENCE_FILE

REFERENCE = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
ENTRY = REFERENCE["entries"][0]
N = REFERENCE["max_new_tokens"]


def wait_for(condition, timeout_s: float = 30.0) -> float:
    """Poll until condition() is true; return how long it took."""
    start = time.monotonic()
    while not condition():
        if time.monotonic() - start > timeout_s:
            raise AssertionError("condition not met in time")
        time.sleep(0.05)
    return time.monotonic() - start


def test_generation_is_reported() -> None:
    with supervised_cluster(two_stage_layout(base_port=50241)) as (pipe, sup):
        infos = pipe.check_layout(n_layer=6)
        assert [i.generation for i in infos] == [1, 1]
        assert [s["generation"] for s in sup.status()] == [1, 1]


def test_crashed_worker_is_restarted_and_serves_again() -> None:
    with supervised_cluster(two_stage_layout(base_port=50243)) as (pipe, sup):
        sup.kill(1)                                                     # worker B dies, like a power cut
        recovered = wait_for(lambda: sup.status()[1]["generation"] == 2 and sup.status()[1]["up"])
        assert recovered < 20
        assert "process exited" in " ".join(what for _, _, what in sup.events)
        new_ids, _ = generate(pipe, "after-crash", ENTRY["prompt_ids"], N)
        assert new_ids == ENTRY["new_ids"]


def test_hung_worker_is_detected_by_missed_heartbeats() -> None:
    """B's process stays alive but stops answering after 3 s: only heartbeats can tell."""
    a, b = two_stage_layout(base_port=50245)
    hung_b = WorkerSpec(b.worker_id, b.port, b.first, b.last, head=True, first_gen_args=("--hang-after-s", "3"))
    with supervised_cluster([a, hung_b], interval_s=0.25, misses_allowed=3) as (pipe, sup):
        wait_for(lambda: sup.status()[1]["generation"] == 2 and sup.status()[1]["up"], timeout_s=30)
        down = [what for _, wid, what in sup.events if wid == "B" and what.startswith("DOWN")]
        assert len(down) == 1 and "missed 3 heartbeats" in down[0]
        new_ids, _ = generate(pipe, "after-hang", ENTRY["prompt_ids"], N)
        assert new_ids == ENTRY["new_ids"]                              # generation 2 has no hang fault


def test_in_flight_request_is_lost_cleanly_then_service_resumes() -> None:
    with supervised_cluster(two_stage_layout(base_port=50247)) as (pipe, sup):
        server, scheduler = start_coordinator(pipe, port=50249, queue_size=4, health=sup)
        channel = grpc.insecure_channel("127.0.0.1:50249")
        stub = pb_grpc.CoordinatorStub(channel)
        try:
            result = {}

            def long_request():
                try:
                    stub.Generate(pb.GenerateRequest(request_id="victim", prompt_ids=ENTRY["prompt_ids"],
                                                     max_new_tokens=200), timeout=30)
                    result["code"] = grpc.StatusCode.OK
                except grpc.RpcError as e:
                    result["code"], result["details"] = e.code(), e.details()

            t = threading.Thread(target=long_request)
            t.start()
            time.sleep(1.0)                                             # mid-generation
            sup.kill(1)
            t.join(timeout=30)
            assert result["code"] == grpc.StatusCode.UNAVAILABLE and "retry" in result["details"]

            reply = stub.Generate(pb.GenerateRequest(request_id="after", prompt_ids=ENTRY["prompt_ids"],
                                                     max_new_tokens=N), timeout=30)
            assert list(reply.new_ids) == ENTRY["new_ids"]              # waited for the restart, then served
            stats = stub.Stats(pb.StatsRequest(), timeout=5)
            assert stats.lost == 1 and stats.completed == 1
            infos = [s.Info(pb.InfoRequest(), timeout=5) for s in pipe.stubs]
            assert [i.cached_requests for i in infos] == [0, 0]          # A's copy of "victim" was released
        finally:
            channel.close()
            server.stop(grace=None)
            scheduler.stop()
