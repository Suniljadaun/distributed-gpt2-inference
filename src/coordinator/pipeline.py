"""
pipeline.py - Chapter 8 (Ch 9 KV-cache; Ch 10 retries; Ch 11 step_batch; Ch 12 reconnect; Ch 14 fencing;
                         Ch 15 layout gate, epochs, step statistics)
The coordinator's view of the workers: send each step through the stages in order,
and generate text token by token, exactly like src/model/generate.py but across processes.
"""
import math
import random
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import grpc
import torch

from src.model.config import config_for
from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc
from src.proto.tensor_codec import tensor_to_proto

N_CTX = 1024
RETRYABLE = {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED}
# Ch 12: after a worker restarts on the same port, reconnect within ~0.5 s (gRPC's default backoff grows to 120 s)
FAST_RECONNECT = [("grpc.initial_reconnect_backoff_ms", 100), ("grpc.min_reconnect_backoff_ms", 100),
                  ("grpc.max_reconnect_backoff_ms", 500)]
# Ch 15: KV-cache migration can send more than gRPC's 4 MB default in one message
CHANNEL_OPTIONS = FAST_RECONNECT + [("grpc.max_send_message_length", 256 << 20),
                                    ("grpc.max_receive_message_length", 256 << 20)]


class LayoutGate:
    """Ch 15: a readers-writer lock. Steps hold it SHARED (many at once); a layout change holds it
    EXCLUSIVE (alone, after every running step has finished). Waiting writers go first, so a busy
    system cannot postpone a layer move forever."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    @contextmanager
    def shared(self) -> Iterator[None]:
        with self._cond:
            while self._writer or self._writers_waiting:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                self._cond.notify_all()

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        with self._cond:
            self._writers_waiting += 1
            while self._writer or self._readers:
                self._cond.wait()
            self._writers_waiting -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


class DeadlineExceeded(Exception):
    """The REQUEST's deadline passed (not just one call's timeout)."""


class StaleReply(Exception):
    """Ch 14: a reply came from a worker generation other than the one we addressed."""


def backoff_delay(attempt: int, base_s: float, cap_s: float) -> float:
    """Exponential backoff with full jitter: random in [0, min(cap, base * 2**attempt)]."""
    return random.uniform(0.0, min(cap_s, base_s * 2 ** attempt))


@dataclass(frozen=True)
class StepTiming:
    total_ms: float               # measured by the coordinator, around the whole step
    compute_ms: tuple[float, ...]  # measured inside each worker
    retries: int = 0              # Ch 10: calls in this step that had to be retried
    epoch: int = 0                # layout epoch the step ran under (0 = no rebalancer, layout fixed)

    @property
    def overhead_ms(self) -> float:
        return self.total_ms - sum(self.compute_ms)   # gRPC + serialisation + Python + retries


class Pipeline:
    def __init__(self, addresses: list[str], timeout_s: float = 10.0, max_attempts: int = 3,
                 backoff_base_s: float = 0.05, backoff_cap_s: float = 1.0) -> None:
        self.addresses = addresses
        self.timeout_s = timeout_s              # longest a single worker call may take
        self.max_attempts = max_attempts        # 1 = never retry
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self.channels = [grpc.insecure_channel(a, options=CHANNEL_OPTIONS) for a in addresses]
        self.stubs = [pb_grpc.StageWorkerStub(c) for c in self.channels]
        self.expected = [0] * len(addresses)     # Ch 14: generation each stage must be (0 = don't check)
        self.epoch = 0                           # Ch 15: layout epoch every call assumes (0 = don't check)
        self.gate = LayoutGate()                 # Ch 15: steps share it; a layer move takes it alone
        # (time, tokens, compute ms per stage, blocks ms per stage), one entry per decode step
        self.stats: deque[tuple[float, int, tuple[float, ...], tuple[float, ...]]] = deque(maxlen=2000)
        self.stats_lock = threading.Lock()

    def _record(self, tokens: int, compute: list[float], blocks: list[float]) -> None:
        """Ch 15: remember each DECODE step's per-stage times, for the rebalancer's monitor."""
        with self.stats_lock:
            self.stats.append((time.monotonic(), tokens, tuple(compute), tuple(blocks)))

    def recent(self, since: float) -> list[tuple[float, int, tuple[float, ...], tuple[float, ...]]]:
        with self.stats_lock:
            return [s for s in self.stats if s[0] >= since]

    def set_expected(self, stage: int, generation: int) -> None:
        """Ch 14: from now on, only talk to (and only believe) this generation of `stage`."""
        self.expected[stage] = generation

    def _check_generation(self, stage: int, reply: pb.ForwardReply) -> None:
        if self.expected[stage] and reply.generation != self.expected[stage]:
            raise StaleReply(f"stage {stage}: reply from generation {reply.generation}, "
                             f"expected {self.expected[stage]}")

    def wait_ready(self, timeout_s: float) -> None:
        for address, channel in zip(self.addresses, self.channels):
            try:
                grpc.channel_ready_future(channel).result(timeout=timeout_s)
            except grpc.FutureTimeoutError:
                raise RuntimeError(f"worker at {address} did not become ready in {timeout_s} s") from None

    def check_layout(self, n_layer: int | None = None) -> list[pb.InfoReply]:
        """The stages must run the same model and cover its layers 0..n_layer-1 exactly once, in order, embed first,
        head last. n_layer=None: the model's own depth (Ch 18: distilgpt2 6, gpt2 12)."""
        infos = [stub.Info(pb.InfoRequest(), timeout=self.timeout_s) for stub in self.stubs]
        models = {info.model for info in infos}
        if len(models) != 1:
            raise RuntimeError(f"workers run different models: {sorted(models)}")
        if n_layer is None:
            n_layer = config_for(models.pop() or "distilgpt2").n_layer
        expected_first = 0
        for i, info in enumerate(infos):
            if info.first_layer != expected_first:
                raise RuntimeError(f"worker {info.worker_id} starts at layer {info.first_layer}, expected {expected_first}")
            if info.has_embed != (i == 0) or info.has_head != (i == len(infos) - 1):
                raise RuntimeError(f"worker {info.worker_id}: embed must be on the first stage, head on the last")
            expected_first = info.last_layer
        if expected_first != n_layer:
            raise RuntimeError(f"stages end at layer {expected_first}, model has {n_layer}")
        return infos

    def _call(self, method, request, deadline: float, what: str):
        """Ch 10, generalised in Ch 11: call `method(request)`, retried on UNAVAILABLE / DEADLINE_EXCEEDED
        with backoff, never past `deadline`. Returns (reply, retries)."""
        retries = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DeadlineExceeded(f"{what}: deadline passed")
            try:
                return method(request, timeout=min(self.timeout_s, remaining)), retries
            except grpc.RpcError as e:
                if e.code() not in RETRYABLE or retries + 1 >= self.max_attempts:
                    raise
                delay = backoff_delay(retries, self.backoff_base_s, self.backoff_cap_s)
                if time.monotonic() + delay >= deadline:
                    raise DeadlineExceeded(f"{what}: no time left to retry") from e
                retries += 1
                time.sleep(delay)

    def _forward(self, stub: pb_grpc.StageWorkerStub, request: pb.ForwardRequest,
                 deadline: float) -> tuple[pb.ForwardReply, int]:
        return self._call(stub.Forward, request, deadline, f"{request.request_id} step {request.step_id}")

    def step(self, request_id: str, step_id: int, new_ids: list[int], start_pos: int,
             deadline: float = math.inf) -> tuple[int, StepTiming]:
        """One step through every stage: only the NEW tokens travel. Returns the next token id."""
        with self.gate.shared():                       # Ch 15: never while the layout is changing
            start, epoch = time.perf_counter(), self.epoch     # fixed while we hold the gate
            message = tensor_to_proto(torch.tensor([new_ids]))
            compute, blocks, retries = [], [], 0
            for k, stub in enumerate(self.stubs):
                request = pb.ForwardRequest(request_id=request_id, step_id=step_id, input=message,
                                            start_pos=start_pos, expect_generation=self.expected[k],
                                            expect_epoch=epoch)
                reply, r = self._forward(stub, request, deadline)
                self._check_generation(k, reply)
                retries += r
                if (reply.request_id, reply.step_id) != (request_id, step_id):
                    raise RuntimeError(f"reply for {reply.request_id}/{reply.step_id}, expected {request_id}/{step_id}")
                if reply.cache_len != start_pos + len(new_ids):
                    raise RuntimeError(f"worker cache holds {reply.cache_len} positions, "
                                       f"expected {start_pos + len(new_ids)}")
                compute.append(reply.compute_ms)
                blocks.append(reply.blocks_ms)
                message = reply.output                 # hidden state passes through untouched
            if len(new_ids) == 1:
                self._record(1, compute, blocks)       # decode steps only: prefill would distort the monitor
            timing = StepTiming((time.perf_counter() - start) * 1000, tuple(compute), retries, epoch)
            return reply.next_token, timing

    def step_batch(self, items: list[tuple[str, int, int, int]],
                   deadline: float = math.inf) -> tuple[list[int], StepTiming]:
        """Ch 11: ONE decode step for several requests. items = (request_id, step_id, last_token, start_pos).
        One ForwardBatch call per stage carries them all. Returns each request's next token, in order."""
        with self.gate.shared():
            start = time.perf_counter()
            inputs = [tensor_to_proto(torch.tensor([[tok]])) for _, _, tok, _ in items]
            compute, blocks, retries = [], [], 0
            for k, stub in enumerate(self.stubs):
                batch = pb.ForwardBatchRequest(items=[
                    pb.ForwardRequest(request_id=rid, step_id=sid, input=x, start_pos=pos,
                                      expect_generation=self.expected[k], expect_epoch=self.epoch)
                    for (rid, sid, _, pos), x in zip(items, inputs)])
                reply, r = self._call(stub.ForwardBatch, batch, deadline, f"batch of {len(items)}")
                retries += r
                for (rid, sid, _, pos), rep in zip(items, reply.items):
                    self._check_generation(k, rep)
                    if (rep.request_id, rep.step_id, rep.cache_len) != (rid, sid, pos + 1):
                        raise RuntimeError(f"batch reply {rep.request_id}/{rep.step_id}/{rep.cache_len} "
                                           f"does not match {rid}/{sid}/{pos + 1}")
                compute.append(reply.compute_ms)
                blocks.append(reply.blocks_ms)
                inputs = [rep.output for rep in reply.items]
            self._record(len(items), compute, blocks)
            timing = StepTiming((time.perf_counter() - start) * 1000, tuple(compute), retries)
            return [rep.next_token for rep in reply.items], timing

    def release(self, request_id: str) -> None:
        """Tell every worker to drop this request's KV-cache. Best effort: never raises."""
        with self.gate.shared():
            for stub in self.stubs:
                try:
                    stub.Release(pb.ReleaseRequest(request_id=request_id), timeout=min(self.timeout_s, 1.0))
                except grpc.RpcError:
                    pass                               # a dead worker has no cache to free anyway

    def close(self) -> None:
        for channel in self.channels:
            channel.close()


def generate(pipe: Pipeline, request_id: str, prompt_ids: list[int], max_new_tokens: int,
             eos_id: int | None = None, deadline: float = math.inf) -> tuple[list[int], list[StepTiming]]:
    """Same contract as src.model.generate.generate, plus per-step timings. Uses the workers' KV-caches.
    `deadline` is a time.monotonic() value; DeadlineExceeded is raised when it passes."""
    new: list[int] = []
    timings: list[StepTiming] = []
    feed, start_pos = list(prompt_ids), 0      # step 0 = whole prompt; later steps = one token each
    try:
        for step_id in range(max_new_tokens):
            if start_pos + len(feed) > N_CTX:
                break
            tok, timing = pipe.step(request_id, step_id, feed, start_pos, deadline)
            new.append(tok)
            timings.append(timing)
            start_pos += len(feed)
            feed = [tok]
            if tok == eos_id:
                break
    finally:
        pipe.release(request_id)               # free the workers' memory, even after an error
    return new, timings
