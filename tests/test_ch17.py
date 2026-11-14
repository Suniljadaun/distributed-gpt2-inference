"""
test_ch17.py - Chapter 17
Workers the coordinator does not own (one container each, restarted by Docker): RemoteHealth must notice a
crash, recognise the restarted worker by its NEW generation, move the fence, and the request must still
finish with the same tokens. Here plain processes play the containers and the test plays Docker.
Run:  python -m pytest tests/test_ch17.py -v
"""
import json
import subprocess
import threading
import time

from src.coordinator.cluster import two_stage_layout
from src.coordinator.remote import remote_cluster
from src.coordinator.replay import generate_logged
from src.coordinator.token_log import TokenLog
from src.model.generate import generate_cached
from src.model.weights import load_pretrained
from src.reference import REFERENCE_FILE

ENTRY = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))["entries"][0]
LONG = 100


def start(spec) -> subprocess.Popen:
    return subprocess.Popen(spec.command(generation=0))             # 0 = "auto": the start time in ms


def test_restart_by_someone_else_is_detected_and_invisible(tmp_path) -> None:
    expected = generate_cached(load_pretrained(), ENTRY["prompt_ids"], LONG)
    specs = two_stage_layout(base_port=50621)
    procs = [start(s) for s in specs]
    try:
        with remote_cluster([s.address for s in specs]) as (pipe, health):
            gen_before = health.status()[1]["generation"]
            assert gen_before > 1_000_000_000_000                   # a timestamp, not 1
            log = TokenLog(tmp_path / "log.jsonl")
            out = {}
            t = threading.Thread(target=lambda: out.setdefault("r", generate_logged(
                pipe, log, "victim", ENTRY["prompt_ids"], LONG, deadline=time.monotonic() + 90, health=health)))
            t.start()
            time.sleep(1.0)
            procs[1].kill()                                         # the container dies ...
            procs[1].wait()
            procs[1] = start(specs[1])                              # ... and "Docker" starts it again
            t.join(timeout=90)
            log.close()
            result = out["r"]
            gen_after = health.status()[1]["generation"]
            assert result.new_ids == expected                       # same tokens as with no failure at all
            assert result.replays == 1
            assert gen_after > gen_before                           # recognised as a new incarnation
            assert pipe.expected[1] == gen_after                    # and the fence moved to it
    finally:
        for p in procs:
            p.kill()
            p.wait()


def test_wait_all_up_does_not_trust_an_old_heartbeat() -> None:
    specs = two_stage_layout(base_port=50625)
    procs = [start(s) for s in specs]
    try:
        with remote_cluster([s.address for s in specs]) as (pipe, health):
            procs[1].kill()
            procs[1].wait()
            assert health.wait_all_up(0) is False                   # right away, before 3 missed heartbeats
    finally:
        for p in procs:
            p.kill()
            p.wait()
