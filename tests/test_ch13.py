"""
test_ch13.py - Chapter 13
The token log survives crashes (torn tail dropped, real corruption detected), and requests
survive worker crashes and coordinator restarts by being replayed from the log, producing
exactly the tokens they would have produced without any failure.
Run:  python -m pytest tests/test_ch13.py -v
"""
import json
import threading
import time

import grpc
import pytest

from src.coordinator.cluster import supervised_cluster, two_stage_layout
from src.coordinator.replay import generate_logged
from src.coordinator.server import start_coordinator
from src.coordinator.token_log import LogCorrupted, TokenLog
from src.model.generate import generate_cached
from src.model.weights import load_pretrained
from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc
from src.reference import REFERENCE_FILE

REFERENCE = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
ENTRY = REFERENCE["entries"][0]
N = REFERENCE["max_new_tokens"]
LONG = 100


@pytest.fixture(scope="module")
def expected_long() -> list[int]:
    """What 100 tokens of the first prompt look like with no failures at all (single process)."""
    return generate_cached(load_pretrained(), ENTRY["prompt_ids"], LONG)


# ---------------- the log on its own ----------------

def test_log_round_trip_and_incomplete_requests(tmp_path) -> None:
    path = tmp_path / "log.jsonl"
    log = TokenLog(path)
    log.start("done", [1, 2], 2)
    log.append_token("done", 7)
    log.append_token("done", 8)
    log.end("done", "completed")
    log.start("half", [3], 5)
    log.append_token("half", 9)
    log.close()

    again = TokenLog(path)                                       # a restarted coordinator
    assert again.get("done").tokens == [7, 8] and again.get("done").status == "completed"
    assert again.get("half").tokens == [9] and again.incomplete() == ["half"]
    assert again.next_lsn == 6 and not again.dropped_torn_tail     # 6 records written: lsn 0..5


def test_torn_last_record_is_dropped(tmp_path) -> None:
    path = tmp_path / "log.jsonl"
    log = TokenLog(path)
    log.start("r", [1], 3)
    log.append_token("r", 5)
    log.close()
    with open(path, "ab") as f:
        f.write(b'{"lsn":2,"type":"token","rid":"r","i":1,"to')    # the crash happened mid-write
    again = TokenLog(path)
    assert again.dropped_torn_tail and again.get("r").tokens == [5]
    again.append_token("r", 6)                                   # and appending works afterwards
    again.close()
    assert TokenLog(path).get("r").tokens == [5, 6]


def test_corruption_in_the_middle_is_detected(tmp_path) -> None:
    path = tmp_path / "log.jsonl"
    log = TokenLog(path)
    log.start("r", [1], 3)
    log.append_token("r", 5)
    log.append_token("r", 6)
    log.close()
    lines = path.read_bytes().split(b"\n")
    lines[1] = lines[1].replace(b'"tok":5', b'"tok":4')         # a flipped value; the checksum disagrees
    path.write_bytes(b"\n".join(lines))
    with pytest.raises(LogCorrupted):
        TokenLog(path)


# ---------------- replay through real workers ----------------

def test_worker_crash_mid_request_is_invisible(tmp_path, expected_long) -> None:
    with supervised_cluster(two_stage_layout(base_port=50291)) as (pipe, sup):
        log = TokenLog(tmp_path / "log.jsonl")
        out = {}
        t = threading.Thread(target=lambda: out.setdefault("r", generate_logged(
            pipe, log, "victim", ENTRY["prompt_ids"], LONG, deadline=time.monotonic() + 60, health=sup)))
        t.start()
        time.sleep(1.0)                                         # ~30 tokens in
        sup.kill(1)
        t.join(timeout=60)
    result = out["r"]
    assert result.new_ids == expected_long                      # as if nothing had happened
    assert result.replays == 1 and result.replayed_tokens > 10 and result.mismatches == 0


def test_resume_after_coordinator_restart(tmp_path) -> None:
    """The old coordinator committed 7 tokens and died; a new one resumes from the log."""
    path = tmp_path / "log.jsonl"
    old = TokenLog(path)
    old.start("resume-me", ENTRY["prompt_ids"], N)
    for tok in ENTRY["new_ids"][:7]:
        old.append_token("resume-me", tok)
    old.close()                                                 # crash: no "end" record

    new = TokenLog(path)
    assert new.incomplete() == ["resume-me"]
    with supervised_cluster(two_stage_layout(base_port=50293)) as (pipe, sup):
        result = generate_logged(pipe, new, "resume-me", ENTRY["prompt_ids"], N, health=sup)
    assert result.new_ids == ENTRY["new_ids"]
    assert result.replayed_tokens == len(ENTRY["prompt_ids"]) + 6   # prompt + 6 tokens rebuilt, not regenerated
    assert new.incomplete() == []


def test_coordinator_survives_worker_crash_and_answers_repeats_from_the_log(tmp_path, expected_long) -> None:
    with supervised_cluster(two_stage_layout(base_port=50295)) as (pipe, sup):
        server, scheduler = start_coordinator(pipe, port=50297, queue_size=4, health=sup,
                                              log=TokenLog(tmp_path / "log.jsonl"))
        channel = grpc.insecure_channel("127.0.0.1:50297")
        stub = pb_grpc.CoordinatorStub(channel)
        request = pb.GenerateRequest(request_id="survivor", prompt_ids=ENTRY["prompt_ids"], max_new_tokens=LONG)
        try:
            out = {}
            t = threading.Thread(target=lambda: out.setdefault("reply", stub.Generate(request, timeout=60)))
            t.start()
            time.sleep(1.0)
            sup.kill(1)
            t.join(timeout=60)
            reply = out["reply"]
            assert list(reply.new_ids) == expected_long and reply.replays == 1
            stats = stub.Stats(pb.StatsRequest(), timeout=5)
            assert stats.lost == 0 and stats.replays == 1 and stats.completed == 1

            again = stub.Generate(request, timeout=5)              # the client retries the same request id
            assert list(again.new_ids) == expected_long             # same answer ...
            assert stub.Stats(pb.StatsRequest(), timeout=5).admitted == stats.admitted   # ... computed 0 times
        finally:
            channel.close()
            server.stop(grace=None)
            scheduler.stop()
