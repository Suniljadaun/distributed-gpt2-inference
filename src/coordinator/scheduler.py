"""
scheduler.py - Chapter 10
Admission control: a BOUNDED queue of jobs in front of the pipeline, served by `slots` threads.
Full queue -> reject at once. Deadline passed while waiting -> drop without running it.
"""
import math
import queue
import threading
import time
from dataclasses import dataclass, field

import grpc

from src.coordinator.pipeline import DeadlineExceeded, Pipeline, generate

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
