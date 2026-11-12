"""
pipeline.py - Chapter 8 (Ch 9: KV-cache on the workers, Release; Ch 10: deadlines, retries with backoff)
The coordinator's view of the workers: send each step through the stages in order,
and generate text token by token, exactly like src/model/generate.py but across processes.
"""
import math
import random
import time
from dataclasses import dataclass

import grpc
import torch

from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc
from src.proto.tensor_codec import tensor_to_proto

N_CTX = 1024
RETRYABLE = {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED}


class DeadlineExceeded(Exception):
    """The REQUEST's deadline passed (not just one call's timeout)."""


def backoff_delay(attempt: int, base_s: float, cap_s: float) -> float:
    """Exponential backoff with full jitter: random in [0, min(cap, base * 2**attempt)]."""
    return random.uniform(0.0, min(cap_s, base_s * 2 ** attempt))


@dataclass(frozen=True)
class StepTiming:
    total_ms: float               # measured by the coordinator, around the whole step
    compute_ms: tuple[float, ...]  # measured inside each worker
    retries: int = 0              # Ch 10: calls in this step that had to be retried

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
        self.channels = [grpc.insecure_channel(a) for a in addresses]
        self.stubs = [pb_grpc.StageWorkerStub(c) for c in self.channels]

    def wait_ready(self, timeout_s: float) -> None:
        for address, channel in zip(self.addresses, self.channels):
            try:
                grpc.channel_ready_future(channel).result(timeout=timeout_s)
            except grpc.FutureTimeoutError:
                raise RuntimeError(f"worker at {address} did not become ready in {timeout_s} s") from None

    def check_layout(self, n_layer: int) -> list[pb.InfoReply]:
        """The stages must cover layers 0..n_layer-1 exactly once, in order, embed first, head last."""
        infos = [stub.Info(pb.InfoRequest(), timeout=self.timeout_s) for stub in self.stubs]
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

    def _forward(self, stub: pb_grpc.StageWorkerStub, request: pb.ForwardRequest,
                 deadline: float) -> tuple[pb.ForwardReply, int]:
        """One worker call, retried on UNAVAILABLE / DEADLINE_EXCEEDED, never past `deadline`."""
        retries = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DeadlineExceeded(f"{request.request_id}: deadline passed at step {request.step_id}")
            try:
                return stub.Forward(request, timeout=min(self.timeout_s, remaining)), retries
            except grpc.RpcError as e:
                if e.code() not in RETRYABLE or retries + 1 >= self.max_attempts:
                    raise
                delay = backoff_delay(retries, self.backoff_base_s, self.backoff_cap_s)
                if time.monotonic() + delay >= deadline:
                    raise DeadlineExceeded(f"{request.request_id}: no time left to retry step {request.step_id}") from e
                retries += 1
                time.sleep(delay)

    def step(self, request_id: str, step_id: int, new_ids: list[int], start_pos: int,
             deadline: float = math.inf) -> tuple[int, StepTiming]:
        """One step through every stage: only the NEW tokens travel. Returns the next token id."""
        start = time.perf_counter()
        message = tensor_to_proto(torch.tensor([new_ids]))
        compute, retries = [], 0
        for stub in self.stubs:
            request = pb.ForwardRequest(request_id=request_id, step_id=step_id, input=message, start_pos=start_pos)
            reply, r = self._forward(stub, request, deadline)
            retries += r
            if (reply.request_id, reply.step_id) != (request_id, step_id):
                raise RuntimeError(f"reply for {reply.request_id}/{reply.step_id}, expected {request_id}/{step_id}")
            if reply.cache_len != start_pos + len(new_ids):
                raise RuntimeError(f"worker cache holds {reply.cache_len} positions, expected {start_pos + len(new_ids)}")
            compute.append(reply.compute_ms)
            message = reply.output                     # hidden state passes through untouched
        timing = StepTiming((time.perf_counter() - start) * 1000, tuple(compute), retries)
        return reply.next_token, timing

    def release(self, request_id: str) -> None:
        """Tell every worker to drop this request's KV-cache. Best effort: never raises."""
        for stub in self.stubs:
            try:
                stub.Release(pb.ReleaseRequest(request_id=request_id), timeout=self.timeout_s)
            except grpc.RpcError:
                pass                                   # a dead worker has no cache to free anyway

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
