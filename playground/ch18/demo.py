"""
demo.py - Chapter 18: the live demo, every scope goal on one screen.
Starts a supervised 2-worker engine (each worker on its own cores), the coordinator with the token log,
and the rebalancer; keeps a steady load running; then walks through:
  1. correctness (G1)        2. live throughput + automatic rebalancing (G2, G4, G6)
  3. kill a worker mid-request: invisible to the client (G5)
  4. a "noisy neighbour" slows worker A: layers move away, then back (G6)
  5. duplicate requests and overload (G3, idempotency)
Run:  python -m playground.ch18.demo            (press Enter between steps)
      python -m playground.ch18.demo --auto     (no pauses, ~2 minutes)
"""
import argparse
import json
import statistics
import tempfile
import threading
import time
from pathlib import Path

import grpc

from src.coordinator.cluster import WorkerSpec, supervised_cluster, two_stage_layout
from src.coordinator.rebalancer import Rebalancer
from src.coordinator.server import start_coordinator
from src.coordinator.token_log import TokenLog
from src.model.generate import generate_cached
from src.model.tokenizer import Tokenizer
from src.model.weights import load_pretrained
from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc
from src.reference import REFERENCE_FILE
from src.worker.server import pin_to_cpus

PORT = 50501
REFERENCE = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
PROMPT = REFERENCE["entries"][0]["prompt_ids"]


class Demo:
    def __init__(self, auto: bool, workdir: Path) -> None:
        self.auto = auto
        self.flag = workdir / "slow_A.flag"
        self.log_path = workdir / "token_log.jsonl"
        self.stop_load = threading.Event()
        self.load_on = threading.Event()                            # cleared = load clients pause
        self.load_on.set()
        self.tok = Tokenizer()

    def pause(self, seconds: float, prompt: str = "") -> None:
        if self.auto:
            time.sleep(seconds)
        else:
            input(f"\n  [Enter] {prompt}")

    @staticmethod
    def title(text: str) -> None:
        print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)

    def layout(self) -> list[WorkerSpec]:
        a, b = two_stage_layout(base_port=PORT)
        common = ("--threads", "2", "--serial")
        return [WorkerSpec(a.worker_id, a.port, a.first, a.last, embed=True,
                           extra_args=common + ("--cpus", "0,1", "--block-delay-ms", "8",
                                                "--block-delay-file", str(self.flag))),
                WorkerSpec(b.worker_id, b.port, b.first, b.last, head=True, extra_args=common + ("--cpus", "2,3"))]

    def status_line(self, pipe, sup, reb, since: float) -> str:
        steps = pipe.recent(since)
        rate = sum(s[1] for s in steps) / max(0.001, time.monotonic() - since)
        a = statistics.median(s[2][0] for s in steps) if steps else 0
        b = statistics.median(s[2][1] for s in steps) if steps else 0
        gens = "/".join(str(s["generation"]) for s in sup.status())
        layout = " | ".join(f"{f}-{l - 1}" for f, l in reb.layout)
        return (f"  {rate:5.1f} tok/s   A {a:5.1f} ms  B {b:5.1f} ms   layers {layout:<9} "
                f"epoch {reb.epoch}  generations {gens}")

    def watch(self, pipe, sup, reb, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            since = time.monotonic()
            time.sleep(2.0)
            print(self.status_line(pipe, sup, reb, since), flush=True)

    def run(self) -> None:
        pin_to_cpus([4, 5])                                         # coordinator + clients: their own cores
        self.flag.unlink(missing_ok=True)
        self.title("Starting: supervisor + 2 workers (own cores, serial stages), coordinator with token log, "
                   "rebalancer")
        expected = generate_cached(load_pretrained(), PROMPT, 60)   # the no-failure answer, single process
        with supervised_cluster(self.layout()) as (pipe, sup):
            log = TokenLog(self.log_path)
            reb = Rebalancer(pipe, supervisor=sup, period_s=1.0, window_s=3.0, cooldown_s=3.0, min_samples=15)
            server, scheduler = start_coordinator(pipe, PORT + 2, queue_size=8, slots=4, health=sup, log=log)
            channel = grpc.insecure_channel(f"127.0.0.1:{PORT + 2}")
            stub = pb_grpc.CoordinatorStub(channel)
            try:
                self.step1(stub)
                self.step2(stub, pipe, sup, reb)
                self.step3(stub, pipe, sup, reb, expected)
                self.step4(pipe, sup, reb)
                self.step5(stub)
            finally:
                self.stop_load.set()
                reb.stop()
                self.flag.unlink(missing_ok=True)
                channel.close()
                server.stop(grace=None)
                scheduler.stop()
                log.close()
        self.title("Demo finished")

    def step1(self, stub) -> None:
        self.title("1. Correctness (G1): the reference prompts through 2 workers, compared token by token")
        for i, entry in enumerate(REFERENCE["entries"]):
            reply = stub.Generate(pb.GenerateRequest(request_id=f"ref-{i}", prompt_ids=entry["prompt_ids"],
                                                     max_new_tokens=REFERENCE["max_new_tokens"]), timeout=30)
            same = list(reply.new_ids) == entry["new_ids"]
            print(f"  {entry['prompt']!r} -> {self.tok.decode(list(reply.new_ids))[:60]!r}... "
                  f"{'MATCHES single-process' if same else 'DIFFERENT'}", flush=True)
        self.pause(1, "start the load")

    def step2(self, stub, pipe, sup, reb) -> None:
        self.title("2. Steady load (8 clients, 4 in flight), KV-cache on every worker; the rebalancer watches")

        def client(c: int) -> None:
            k = 0
            while not self.stop_load.is_set():
                self.load_on.wait()
                try:
                    stub.Generate(pb.GenerateRequest(request_id=f"load-{c}-{k}", prompt_ids=PROMPT,
                                                     max_new_tokens=30), timeout=60)
                except grpc.RpcError:
                    time.sleep(0.1)                                 # rejected or lost: just try again
                k += 1

        for c in range(8):
            threading.Thread(target=client, args=(c,), daemon=True).start()
        reb.start()
        self.watch(pipe, sup, reb, 12)
        self.pause(1, "kill worker B")

    def step3(self, stub, pipe, sup, reb, expected: list[int]) -> None:
        self.title("3. Crash (G5): worker B is killed while a 60-token request is running")
        self.load_on.clear()                                        # pause the load and let the coordinator
        drain_until = time.monotonic() + 20                         # drain, so the watched request runs at
        while time.monotonic() < drain_until:                       # once instead of waiting in the queue
            s = stub.Stats(pb.StatsRequest(), timeout=5)
            if s.queued == 0 and s.running == 0:
                break
            time.sleep(0.1)
        out = {}
        t = threading.Thread(target=lambda: out.setdefault("r", stub.Generate(pb.GenerateRequest(
            request_id="watched", prompt_ids=PROMPT, max_new_tokens=60), timeout=60)))
        start = time.monotonic()
        t.start()
        time.sleep(0.6)
        print("  >>> killing worker B", flush=True)
        sup.kill(1)
        t.join()
        r = out["r"]
        print(f"  the watched request finished in {time.monotonic() - start:.1f} s: replays {r.replays}, "
              f"identical to the no-failure answer: {list(r.new_ids) == expected}", flush=True)
        self.load_on.set()
        self.watch(pipe, sup, reb, 6)
        self.pause(1, "slow worker A down")

    def step4(self, pipe, sup, reb) -> None:
        self.title("4. Rebalancing (G6): a noisy neighbour makes worker A slower (+8 ms per block)")
        self.flag.touch()
        print(f"  >>> created {self.flag.name}: A is now slow", flush=True)
        self.watch(pipe, sup, reb, 16)
        self.flag.unlink()
        print(f"  >>> deleted {self.flag.name}: A is fast again", flush=True)
        self.watch(pipe, sup, reb, 12)
        self.pause(1, "duplicates and overload")

    def step5(self, stub) -> None:
        self.title("5. Duplicates and overload (G3, idempotency)")
        before = stub.Stats(pb.StatsRequest(), timeout=5)
        replies = []
        threads = [threading.Thread(target=lambda: replies.append(stub.Generate(pb.GenerateRequest(
            request_id="same-id", prompt_ids=PROMPT, max_new_tokens=20), timeout=60))) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        again = stub.Generate(pb.GenerateRequest(request_id="same-id", prompt_ids=PROMPT, max_new_tokens=20), timeout=5)
        after = stub.Stats(pb.StatsRequest(), timeout=5)
        same = all(list(r.new_ids) == list(again.new_ids) for r in replies)
        print(f"  3 concurrent copies + 1 later retry of 'same-id': identical answers {same}, "
              f"coalesced {after.coalesced - before.coalesced}, the retry answered from the log", flush=True)
        print(f"  coordinator totals: admitted {after.admitted}, completed {after.completed}, "
              f"rejected (queue full) {after.rejected}, replays {after.replays}, lost {after.lost}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="no pauses")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as d:
        Demo(args.auto, Path(d)).run()


if __name__ == "__main__":
    main()
