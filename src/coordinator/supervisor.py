"""
supervisor.py - Chapter 12 (Ch 14: listeners told about every new generation; Ch 15: update_spec)
Owns the worker processes. A monitor thread sends a heartbeat (Ping) to every worker every
`interval_s`. A worker is declared DEAD when its process has exited, or when it misses
`misses_allowed` heartbeats in a row (alive but hung). A dead worker is killed, restarted on the
same port with generation + 1, and marked UP only after it answers a Ping as the new generation.
"""
import math
import subprocess
import threading
import time
from dataclasses import dataclass, field, replace

import grpc

from src.coordinator.cluster import WorkerSpec
from src.coordinator.pipeline import FAST_RECONNECT
from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc


@dataclass
class WorkerHandle:
    spec: WorkerSpec
    stub: pb_grpc.StageWorkerStub
    proc: subprocess.Popen | None = None
    generation: int = 0
    up: bool = False
    misses: int = 0                   # heartbeats missed in a row
    restarts: int = 0
    last_ok: float = field(default_factory=time.monotonic)


class Supervisor:
    def __init__(self, layout: list[WorkerSpec], interval_s: float = 0.25, misses_allowed: int = 3,
                 ping_timeout_s: float = 0.25, ready_timeout_s: float = 60.0) -> None:
        self.interval_s = interval_s
        self.misses_allowed = misses_allowed
        self.ping_timeout_s = ping_timeout_s
        self.ready_timeout_s = ready_timeout_s
        self.channels = [grpc.insecure_channel(s.address, options=FAST_RECONNECT) for s in layout]
        self.workers = [WorkerHandle(s, pb_grpc.StageWorkerStub(c)) for s, c in zip(layout, self.channels)]
        self.lock = threading.Lock()
        self.events: list[tuple[float, str, str]] = []          # (time, worker id, what happened)
        self.all_up = threading.Event()
        self.listeners: list = []                               # Ch 14: called as fn(index, generation) on UP
        self._stop = threading.Event()
        self._monitor = threading.Thread(target=self._watch, daemon=True)

    # ---------------- life cycle ----------------

    def start(self) -> None:
        for w in self.workers:
            self._launch(w)
        for w in self.workers:
            self._wait_ready(w)
        self.all_up.set()
        self._monitor.start()

    def stop(self) -> None:
        self._stop.set()
        if self._monitor.is_alive():
            self._monitor.join()
        for w in self.workers:
            if w.proc is not None and w.proc.poll() is None:
                w.proc.kill()
                w.proc.wait()
        for c in self.channels:
            c.close()

    def wait_all_up(self, timeout_s: float) -> bool:
        """Block until every worker is UP (True) or timeout_s passes (False). timeout_s may be math.inf.
        Ch 18 fix: a caller whose RPC just failed may get here BEFORE the monitor's next sweep has seen the
        crash; all_up would still be set and it would retry at once against the dead port (and burn a replay).
        So first look at the processes ourselves: an exited one is marked DOWN now; the monitor restarts it."""
        self._notice_exits()
        return self.all_up.wait(None if math.isinf(timeout_s) else max(0.0, timeout_s))

    def _notice_exits(self) -> None:
        with self.lock:
            for w in self.workers:
                if w.up and w.proc is not None and w.proc.poll() is not None:
                    w.up = False
                    self.all_up.clear()

    def update_spec(self, index: int, **changes) -> None:
        """Ch 15: the layout changed; a restart of this worker must use the new range and epoch."""
        with self.lock:
            w = self.workers[index]
            w.spec = replace(w.spec, **changes)

    def kill(self, index: int) -> None:
        """Test helper: crash a worker the hard way, like a power cut."""
        self.workers[index].proc.kill()

    def status(self) -> list[dict]:
        with self.lock:
            return [{"worker": w.spec.worker_id, "up": w.up, "generation": w.generation,
                     "restarts": w.restarts, "pid": w.proc.pid if w.proc else None} for w in self.workers]

    # ---------------- internals ----------------

    def _log(self, w: WorkerHandle, what: str) -> None:
        with self.lock:
            self.events.append((time.monotonic(), w.spec.worker_id, what))
        print(f"    [supervisor] worker {w.spec.worker_id}: {what}", flush=True)

    def _launch(self, w: WorkerHandle) -> None:
        w.generation += 1
        w.proc = subprocess.Popen(w.spec.command(w.generation))
        self._log(w, f"started generation {w.generation} (pid {w.proc.pid})")

    def _ping(self, w: WorkerHandle) -> bool:
        """One heartbeat: True only if THIS generation answered in time."""
        try:
            reply = w.stub.Ping(pb.PingRequest(), timeout=self.ping_timeout_s)
        except grpc.RpcError:
            return False
        return reply.generation == w.generation

    def _wait_ready(self, w: WorkerHandle) -> None:
        deadline = time.monotonic() + self.ready_timeout_s
        while not self._ping(w):
            if w.proc.poll() is not None:
                raise RuntimeError(f"worker {w.spec.worker_id} generation {w.generation} exited during start-up")
            if time.monotonic() > deadline:
                raise RuntimeError(f"worker {w.spec.worker_id} not ready after {self.ready_timeout_s} s")
            time.sleep(0.05)
        for listener in self.listeners:                         # Ch 14: e.g. the pipeline's fencing token
            listener(self.workers.index(w), w.generation)
        with self.lock:
            w.up, w.misses, w.last_ok = True, 0, time.monotonic()
        self._log(w, f"UP as generation {w.generation}")

    def _watch(self) -> None:
        while not self._stop.wait(self.interval_s):
            for w in self.workers:
                if self._stop.is_set():
                    return
                exited = w.proc.poll() is not None
                if not exited and self._ping(w):
                    w.misses, w.last_ok = 0, time.monotonic()
                    continue
                w.misses += 1
                if exited:
                    self._restart(w, f"process exited with code {w.proc.returncode}")
                elif w.misses >= self.misses_allowed:
                    self._restart(w, f"missed {w.misses} heartbeats ({time.monotonic() - w.last_ok:.2f} s silent)")

    def _restart(self, w: WorkerHandle, reason: str) -> None:
        with self.lock:
            w.up = False
        self.all_up.clear()
        self._log(w, f"DOWN: {reason}")
        if w.proc.poll() is None:                     # hung, not dead: kill it so the port is free
            w.proc.kill()
        w.proc.wait()
        self._launch(w)
        self._wait_ready(w)
        with self.lock:
            w.restarts += 1
            if all(x.up for x in self.workers):
                self.all_up.set()
