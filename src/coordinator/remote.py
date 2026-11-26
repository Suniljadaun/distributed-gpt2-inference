"""
remote.py - Chapter 17
Health checking for workers the coordinator does NOT own: e.g. one Docker container per worker, restarted
by Docker's restart policy instead of by our Supervisor. It offers the same interface as the Supervisor
(wait_all_up, listeners, status), but learns everything by pinging:
  - a worker is DOWN after `misses_allowed` failed pings in a row;
  - a new incarnation is recognised by a NEW generation in its Ping reply (workers started with
    --generation 0 use their start time, so every restart is a larger number) and the listeners are told,
    which moves the pipeline's fence (Ch 14) to the new generation;
  - wait_all_up() first pings everyone itself (the Ch 18 lesson: don't trust a heartbeat that may be up to
    one interval old when a call has just failed).
"""
import math
import threading
import time
from contextlib import contextmanager
from typing import Iterator

import grpc

from src.coordinator.pipeline import FAST_RECONNECT, Pipeline
from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc


class RemoteHealth:
    def __init__(self, addresses: list[str], interval_s: float = 0.25, misses_allowed: int = 3,
                 ping_timeout_s: float = 0.5) -> None:
        self.addresses = addresses
        self.interval_s, self.misses_allowed, self.ping_timeout_s = interval_s, misses_allowed, ping_timeout_s
        self.channels = [grpc.insecure_channel(a, options=FAST_RECONNECT) for a in addresses]
        self.stubs = [pb_grpc.StageWorkerStub(c) for c in self.channels]
        self.generation = [0] * len(addresses)          # last generation seen per worker (0 = never)
        self.up = [False] * len(addresses)
        self.misses = [0] * len(addresses)
        self.lock = threading.Lock()
        self.all_up = threading.Event()
        self.listeners: list = []                       # fn(index, generation), called on every new generation
        self.events: list[tuple[float, str, str]] = []
        self._stop = threading.Event()
        self._monitor = threading.Thread(target=self._watch, daemon=True)

    # ---------------- life cycle ----------------

    def start(self, ready_timeout_s: float = 300.0) -> None:
        """Wait until every worker answers (containers may still be loading the model), then watch them."""
        deadline = time.monotonic() + ready_timeout_s
        while not self._check_all():
            if time.monotonic() > deadline:
                down = [a for a, u in zip(self.addresses, self.up) if not u]
                raise RuntimeError(f"workers not reachable after {ready_timeout_s} s: {down}")
            time.sleep(0.5)
        self._monitor.start()

    def stop(self) -> None:
        self._stop.set()
        if self._monitor.is_alive():
            self._monitor.join()
        for c in self.channels:
            c.close()

    # ---------------- the Supervisor-compatible interface ----------------

    def wait_all_up(self, timeout_s: float) -> bool:
        if self._check_all():                           # look now, not at the last heartbeat
            return True
        return self.all_up.wait(None if math.isinf(timeout_s) else max(0.0, timeout_s))

    def status(self) -> list[dict]:
        with self.lock:
            return [{"worker": a, "up": u, "generation": g}
                    for a, u, g in zip(self.addresses, self.up, self.generation)]

    # ---------------- internals ----------------

    def _ping(self, i: int) -> int | None:
        try:
            return self.stubs[i].Ping(pb.PingRequest(), timeout=self.ping_timeout_s).generation
        except grpc.RpcError:
            return None

    def _observe(self, i: int, generation: int | None, strict: bool) -> None:
        """Record one ping result. strict = one failure is enough to mark the worker down."""
        new_generation = False
        with self.lock:
            if generation is None:
                self.misses[i] += 1
                if self.up[i] and (strict or self.misses[i] >= self.misses_allowed):
                    self.up[i] = False
                    self._log(i, f"DOWN ({self.misses[i]} failed pings)")
            else:
                self.misses[i] = 0
                if generation != self.generation[i]:
                    new_generation = True
                    self.generation[i] = generation
                if not self.up[i] or new_generation:
                    self._log(i, f"UP as generation {generation}")
                self.up[i] = True
            if all(self.up):
                self.all_up.set()
            else:
                self.all_up.clear()
        if new_generation:
            for listener in self.listeners:            # outside the lock: listeners may call back into us
                listener(i, generation)

    def _check_all(self) -> bool:
        for i in range(len(self.addresses)):
            self._observe(i, self._ping(i), strict=True)
        return self.all_up.is_set()

    def _watch(self) -> None:
        while not self._stop.wait(self.interval_s):
            for i in range(len(self.addresses)):
                self._observe(i, self._ping(i), strict=False)

    def _log(self, i: int, what: str) -> None:            # called with self.lock held
        self.events.append((time.monotonic(), self.addresses[i], what))
        print(f"    [health] worker {self.addresses[i]}: {what}", flush=True)


@contextmanager
def remote_cluster(addresses: list[str], **options) -> Iterator[tuple[Pipeline, RemoteHealth]]:
    """Like supervised_cluster, for workers that are already running somewhere else (addresses in stage order)."""
    health = RemoteHealth(addresses, **options)
    health.start()
    pipe = Pipeline(addresses, timeout_s=2.0)
    for i, s in enumerate(health.status()):             # Ch 14 fencing: only believe the current generation,
        pipe.set_expected(i, s["generation"])           # and follow every restart
    health.listeners.append(pipe.set_expected)
    try:
        pipe.check_layout()
        yield pipe, health
    finally:
        pipe.close()
        health.stop()
