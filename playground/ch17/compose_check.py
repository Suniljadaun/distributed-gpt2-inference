"""
compose_check.py - Chapter 17: check the Docker Compose deployment from the host.
  1. G1 across containers: the reference prompts through the containerised coordinator, token by token.
  2. G5 across containers: kill the worker process inside the worker-b container in the middle of a
     100-token request; Docker's restart policy starts a new container, the coordinator notices the new
     generation and replays the log; the answer must equal single-process inference.
Run (after `docker compose up -d --build`):  python -m playground.ch17.compose_check
"""
import json
import subprocess
import threading
import time

import grpc

from src.model.generate import generate_cached
from src.model.weights import load_pretrained
from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc
from src.reference import REFERENCE_FILE

ADDRESS = "127.0.0.1:50060"
REFERENCE = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
PROMPT = REFERENCE["entries"][0]["prompt_ids"]


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def restart_count(service: str) -> int:
    cid = sh("docker", "compose", "ps", "-q", service)
    return int(sh("docker", "inspect", "--format", "{{.RestartCount}}", cid))


def main() -> None:
    channel = grpc.insecure_channel(ADDRESS)
    grpc.channel_ready_future(channel).result(timeout=120)          # the coordinator may still be starting
    stub = pb_grpc.CoordinatorStub(channel)
    print("containers:\n" + sh("docker", "compose", "ps", "--format", "  {{.Service}}  {{.Status}}"))

    print("\n1. Correctness across containers (G1)")
    ok = True
    for i, e in enumerate(REFERENCE["entries"]):
        r = stub.Generate(pb.GenerateRequest(request_id=f"compose-ref-{time.time_ns()}-{i}", prompt_ids=e["prompt_ids"],
                                             max_new_tokens=REFERENCE["max_new_tokens"]), timeout=120)
        ok &= list(r.new_ids) == e["new_ids"]
        print(f"  {e['prompt']!r}: {'MATCHES single-process' if list(r.new_ids) == e['new_ids'] else 'DIFFERENT'}")

    print("\n2. Crash inside a container (G5)")
    expected = generate_cached(load_pretrained(), PROMPT, 100)
    restarts_before = restart_count("worker-b")
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", stub.Generate(pb.GenerateRequest(
        request_id=f"compose-crash-{time.time_ns()}", prompt_ids=PROMPT, max_new_tokens=100), timeout=180)))
    start = time.monotonic()
    t.start()
    time.sleep(1.5)
    print("  >>> killing the worker process in container worker-b (kill -9)")
    subprocess.run(["docker", "compose", "exec", "-T", "worker-b", "pkill", "-9", "-f", "src.worker.server"], check=False)
    t.join()
    r = out["r"]
    print(f"  request finished in {time.monotonic() - start:.1f} s, replays {r.replays}, "
          f"identical to single-process inference: {list(r.new_ids) == expected}")
    print(f"  Docker restarted worker-b: restart count {restarts_before} -> {restart_count('worker-b')}")
    s = stub.Stats(pb.StatsRequest(), timeout=5)
    print(f"  coordinator: completed {s.completed}, replays {s.replays}, lost {s.lost}")
    channel.close()
    ok &= list(r.new_ids) == expected
    if not ok:                                                      # so that CI fails on a wrong answer
        raise SystemExit("compose check FAILED: an output differs from single-process inference")
    print("\ncompose check passed")


if __name__ == "__main__":
    main()
