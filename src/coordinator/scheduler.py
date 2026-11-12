"""
scheduler.py - Chapter 10 (Chapter 11: BatchingScheduler)
Admission control: a BOUNDED queue of jobs in front of the pipeline, served by `slots` threads.
Full queue -> reject at once. Deadline passed while waiting -> drop without running it.
Ch 11: BatchingScheduler runs many requests per thread as one micro-batch (continuous batching).
"""
import math
import queue
import threading
import time
from dataclasses import dataclass, field

import grpc

from src.coordinator.pipeline import N_CTX, DeadlineExceeded, Pipeline, generate

STOP = None


@dataclass
class Job:
    request_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    deadline: float = math.inf                                  # time.monotonic() value
    enqueued: float = field(default_factory=time.monotonic)
    started: float | None = None
    finished: float | None = None
    new_ids: list[int] = field(default_factory=list)
    retries: int = 0
    error: tuple[grpc.StatusCode, str] | None = None
    done: threading.Event = field(default_factory=threading.Event)


class Scheduler:
    def __init__(self, pipe: Pipeline, queue_size: int, slots: int = 1, eos_id: int | None = None) -> None:
        self.pipe = pipe
        self.queue_size = queue_size
        self.eos_id = eos_id
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.lock = threading.Lock()
        self.counts = {"admitted": 0, "rejected": 0, "completed": 0, "expired": 0, "failed": 0, "retries": 0}
        self.running = 0
        self.threads = [threading.Thread(target=self._serve, daemon=True) for _ in range(slots)]

    def start(self) -> None:
        for t in self.threads:
            t.start()

    def stop(self) -> None:
        for _ in self.threads:
            self.queue.put(STOP)                                 # one STOP per thread (Ch 1's sentinel)
        for t in self.threads:
            t.join()

    def submit(self, job: Job) -> bool:
        """Admit the job, or return False at once if the queue is full (load shedding)."""
        try:
            self.queue.put_nowait(job)
        except queue.Full:
            self._count("rejected")
            return False
        self._count("admitted")
        return True

    def stats(self) -> dict[str, int]:
        with self.lock:
            return dict(self.counts, queued=self.queue.qsize(), running=self.running)

    def _count(self, name: str, n: int = 1) -> None:
        with self.lock:
            self.counts[name] += n

    def _serve(self) -> None:
        while (job := self.queue.get()) is not STOP:
            if time.monotonic() >= job.deadline:                 # nobody is waiting for it any more
                job.error = (grpc.StatusCode.DEADLINE_EXCEEDED, "deadline passed while queued")
                self._finish(job, "expired")
                continue
            with self.lock:
                self.running += 1
            job.started = time.monotonic()
            outcome = "completed"
            try:
                job.new_ids, timings = generate(self.pipe, job.request_id, job.prompt_ids,
                                                job.max_new_tokens, self.eos_id, job.deadline)
                job.retries = sum(t.retries for t in timings)
            except DeadlineExceeded as e:
                job.error, outcome = (grpc.StatusCode.DEADLINE_EXCEEDED, str(e)), "expired"
            except grpc.RpcError as e:
                job.error, outcome = (e.code(), f"worker error: {e.details()}"), "failed"
            except Exception as e:                               # a bug must not kill the serving thread
                job.error, outcome = (grpc.StatusCode.INTERNAL, f"{type(e).__name__}: {e}"), "failed"
            with self.lock:
                self.running -= 1
            self._count("retries", job.retries)
            self._finish(job, outcome)

    def _finish(self, job: Job, outcome: str) -> None:
        job.finished = time.monotonic()
        self._count(outcome)
        job.done.set()                                           # wake the gRPC thread waiting for it


@dataclass
class Active:
    """Ch 11: a request inside a running micro-batch."""
    job: Job
    step_id: int                  # the next step to send
    token: int                    # the token that step will feed
    start_pos: int                # its position


class BatchingScheduler(Scheduler):
    """Ch 11: continuous batching. Each of `groups` threads keeps up to `max_batch` requests in flight:
    new requests are prefilled and JOIN between steps, finished ones LEAVE, and every step is one
    ForwardBatch call per stage for the whole group."""

    def __init__(self, pipe: Pipeline, queue_size: int, max_batch: int = 8, groups: int = 1,
                 eos_id: int | None = None) -> None:
        super().__init__(pipe, queue_size, slots=groups, eos_id=eos_id)
        self.max_batch = max_batch
        self.threads = [threading.Thread(target=self._serve_batches, daemon=True) for _ in range(groups)]

    def _serve_batches(self) -> None:
        active: list[Active] = []
        stopping = False
        while active or not stopping:
            while not stopping and len(active) < self.max_batch:           # 1. let new requests join
                try:
                    job = self.queue.get(block=not active)                 # wait only when idle
                except queue.Empty:
                    break
                if job is STOP:
                    stopping = True
                elif time.monotonic() >= job.deadline:
                    job.error = (grpc.StatusCode.DEADLINE_EXCEEDED, "deadline passed while queued")
                    self._finish(job, "expired")
                else:
                    joined = self._prefill(job)
                    if joined is not None:
                        active.append(joined)
            if active:
                active = self._decode_step(active)                         # 2. one step for all of them

    def _prefill(self, job: Job) -> Active | None:
        """Step 0 (the whole prompt) on its own; returns None if the request is already finished."""
        with self.lock:
            self.running += 1
        job.started = time.monotonic()
        try:
            tok, timing = self.pipe.step(job.request_id, 0, job.prompt_ids, 0, job.deadline)
        except Exception as e:
            self._leave(job, *self._outcome(e))
            return None
        job.new_ids.append(tok)
        job.retries += timing.retries
        if job.max_new_tokens == 1 or tok == self.eos_id:
            self._leave(job, "completed")
            return None
        return Active(job, step_id=1, token=tok, start_pos=len(job.prompt_ids))

    def _decode_step(self, active: list[Active]) -> list[Active]:
        """One ForwardBatch step for the whole micro-batch; returns the requests that continue."""
        try:
            tokens, timing = self.pipe.step_batch(
                [(a.job.request_id, a.step_id, a.token, a.start_pos) for a in active])
        except Exception as e:                                             # the whole batch failed
            outcome, error = self._outcome(e)
            for a in active:
                self._leave(a.job, outcome, error)
            return []
        self._count("retries", timing.retries)
        still = []
        for a, tok in zip(active, tokens):
            a.job.new_ids.append(tok)
            a.step_id, a.token, a.start_pos = a.step_id + 1, tok, a.start_pos + 1
            if len(a.job.new_ids) >= a.job.max_new_tokens or tok == self.eos_id or a.start_pos + 1 > N_CTX:
                self._leave(a.job, "completed")
            elif time.monotonic() >= a.job.deadline:
                self._leave(a.job, "expired", (grpc.StatusCode.DEADLINE_EXCEEDED, "deadline passed while running"))
            else:
                still.append(a)
        return still

    @staticmethod
    def _outcome(e: Exception) -> tuple[str, tuple[grpc.StatusCode, str]]:
        if isinstance(e, DeadlineExceeded):
            return "expired", (grpc.StatusCode.DEADLINE_EXCEEDED, str(e))
        if isinstance(e, grpc.RpcError):
            return "failed", (e.code(), f"worker error: {e.details()}")
        return "failed", (grpc.StatusCode.INTERNAL, f"{type(e).__name__}: {e}")

    def _leave(self, job: Job, outcome: str, error: tuple[grpc.StatusCode, str] | None = None) -> None:
        self.pipe.release(job.request_id)
        job.error = error
        with self.lock:
            self.running -= 1
        self._finish(job, outcome)
