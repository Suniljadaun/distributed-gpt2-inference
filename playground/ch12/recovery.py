"""
recovery.py - Chapter 12, Measure it
1. Crash vs hang: how long until the failure is DETECTED, and how long the RESTART takes,
   for three heartbeat intervals.
2. False positives: does an over-eager failure detector kill a worker that is merely busy?
Run:  python -m playground.ch12.recovery            (about 2 minutes)
"""
import threading
import time

from src.coordinator.cluster import WorkerSpec, supervised_cluster, two_stage_layout
from src.coordinator.pipeline import generate

PROMPT = [464, 3139, 286, 4881, 318]


def event_time(sup, worker: str, prefix: str, after: float = 0.0) -> float:
    for t, wid, what in sup.events:
        if wid == worker and what.startswith(prefix) and t >= after:
            return t
    raise RuntimeError(f"no {prefix!r} event for {worker}")


def wait_generation(sup, index: int, gen: int, timeout_s: float = 60) -> None:
    end = time.monotonic() + timeout_s
    while not (sup.status()[index]["generation"] == gen and sup.status()[index]["up"]):
        if time.monotonic() > end:
            raise RuntimeError("worker did not come back")
        time.sleep(0.02)


def crash_and_hang(interval: float, port: int) -> tuple[float, float, float, float]:
    # crash: we kill B ourselves, so we know exactly when it died
    with supervised_cluster(two_stage_layout(base_port=port), interval_s=interval, ping_timeout_s=interval) as (_, sup):
        t_kill = time.monotonic()
        sup.kill(1)
        wait_generation(sup, 1, 2)
        down, up = event_time(sup, "B", "DOWN", t_kill), event_time(sup, "B", "UP as generation 2")
        crash = (down - t_kill, up - down)
    # hang: B stops answering after 3 s; it "dies" just after its last answered heartbeat
    a, b = two_stage_layout(base_port=port + 2)
    hung = WorkerSpec(b.worker_id, b.port, b.first, b.last, head=True, first_gen_args=("--hang-after-s", "3"))
    with supervised_cluster([a, hung], interval_s=interval, ping_timeout_s=interval) as (_, sup):
        wait_generation(sup, 1, 2)
        down, up = event_time(sup, "B", "DOWN"), event_time(sup, "B", "UP as generation 2")
        silent = float([w for _, wid, w in sup.events if wid == "B" and w.startswith("DOWN")][0]
                       .split("(")[1].split(" s")[0])
        hang = (silent, up - down)
    return crash + hang


def false_positives(interval: float, misses: int, timeout: float, port: int, seconds: float = 12) -> int:
    """Keep both workers busy with generation, and count restarts of perfectly healthy workers."""
    with supervised_cluster(two_stage_layout(base_port=port), interval_s=interval, misses_allowed=misses,
                            ping_timeout_s=timeout) as (pipe, sup):
        stop = time.monotonic() + seconds

        def busy(k: int) -> None:
            n = 0
            while time.monotonic() < stop:
                try:
                    generate(pipe, f"busy-{k}-{n}", PROMPT, 40)
                except Exception:
                    pass                                     # a false "death" kills requests too
                n += 1

        threads = [threading.Thread(target=busy, args=(k,)) for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return sum(s["restarts"] for s in sup.status())


def main() -> None:
    print("1. Time to detect and to restart (3 missed heartbeats = dead)")
    print(f"{'interval':>9} | {'crash: detect':>13} {'restart':>8} | {'hang: detect':>12} {'restart':>8}")
    for k, interval in enumerate([0.1, 0.25, 0.5]):
        cd, cr, hd, hr = crash_and_hang(interval, 50251 + 4 * k)
        print(f"{interval:>8.2f}s | {cd:>12.2f}s {cr:>7.2f}s | {hd:>11.2f}s {hr:>7.2f}s")

    print("\n2. False positives: 4 busy clients for 12 s, nothing actually fails")
    print(f"{'detector':<40} {'needless restarts':>18}")
    for k, (label, interval, misses, timeout) in enumerate([
            ("sensible: every 0.25 s, 3 misses, 250 ms", 0.25, 3, 0.25),
            ("over-eager: every 0.1 s, 1 miss, 2 ms", 0.1, 1, 0.002)]):
        print(f"{label:<40} {false_positives(interval, misses, timeout, 50271 + 2 * k):>18}")


if __name__ == "__main__":
    main()
