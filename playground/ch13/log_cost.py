"""
log_cost.py - Chapter 13, Measure it
1. What does committing a token to the log cost: flush (survives a process crash) vs fsync (survives power loss)?
2. What does a replay cost, compared with generating the same tokens again one by one?
3. End to end: a 100-token request with and without a worker crash in the middle.
Run:  python -m playground.ch13.log_cost            (about 1 minute)
"""
import statistics
import tempfile
import threading
import time
from pathlib import Path

from src.coordinator.cluster import running_cluster, supervised_cluster, two_stage_layout
from src.coordinator.replay import generate_logged
from src.coordinator.token_log import TokenLog

PROMPT = [464, 3139, 286, 4881, 318]


def append_cost(sync: str, n: int = 2000) -> tuple[float, float]:
    with tempfile.TemporaryDirectory() as d:
        log = TokenLog(Path(d) / "log.jsonl", sync=sync)
        log.start("r", PROMPT, n)
        t0 = time.perf_counter()
        for i in range(n):
            log.append_token("r", 1000 + i)
        us = (time.perf_counter() - t0) / n * 1e6
        log.close()
        size = (Path(d) / "log.jsonl").stat().st_size
    return us, size / n


def main() -> None:
    print("1. Committing one token to the log")
    for sync in ("flush", "fsync"):
        us, bytes_per = append_cost(sync, 2000 if sync == "flush" else 300)
        print(f"  sync={sync:<6} {us:8.1f} us per token   {bytes_per:5.1f} bytes per token on disk")

    print("\n2. Rebuilding a cache: one replay prefill vs generating the tokens again")
    with running_cluster(two_stage_layout(base_port=50301)) as pipe:
        _, t = pipe.step("probe", 0, PROMPT, 0)
        decode_ms = statistics.median(pipe.step("probe", s, [262], 4 + s)[1].total_ms for s in range(1, 21))
        pipe.release("probe")
        print(f"  (one decode step = {decode_ms:.1f} ms)")
        print(f"  {'tokens to rebuild':>18} {'replay ms':>10} {'regenerate ms (est.)':>21} {'replay is':>10}")
        for length in [25, 100, 400, 800]:
            prefix = [1000 + (i % 3000) for i in range(length)]
            times = []
            for k in range(3):
                _, timing = pipe.step(f"replay-{length}-{k}", 0, prefix, 0)
                pipe.release(f"replay-{length}-{k}")
                times.append(timing.total_ms)
            replay = statistics.median(times)
            regen = length * decode_ms
            print(f"  {length:>18} {replay:>10.1f} {regen:>21.0f} {regen / replay:>8.0f}x faster")

    print("\n3. A 100-token request, end to end")
    with tempfile.TemporaryDirectory() as d, supervised_cluster(two_stage_layout(base_port=50303)) as (pipe, sup):
        log = TokenLog(Path(d) / "log.jsonl")
        t0 = time.monotonic()
        generate_logged(pipe, log, "calm", PROMPT, 100, health=sup)
        calm = time.monotonic() - t0

        out = {}
        t0 = time.monotonic()
        t = threading.Thread(target=lambda: out.setdefault("r", generate_logged(
            pipe, log, "stormy", PROMPT, 100, health=sup)))
        t.start()
        time.sleep(1.0)
        sup.kill(1)
        t.join()
        stormy = time.monotonic() - t0
        r = out["r"]
        same = r.new_ids == log.get("calm").tokens
        log.close()                                    # Windows cannot delete an open file
        print(f"  no failure          : {calm:5.2f} s")
        print(f"  worker B killed at 1 s: {stormy:5.2f} s  (replays {r.replays}, rebuilt {r.replayed_tokens} positions, "
              f"mismatches {r.mismatches}, identical output: {same})")


if __name__ == "__main__":
    main()
