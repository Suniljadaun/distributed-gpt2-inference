"""
pipeline.py - Chapter 8
The coordinator's view of the workers: send each step through the stages in order,
and generate text token by token, exactly like src/model/generate.py but across processes.
"""
import time
from dataclasses import dataclass

import grpc
import torch

from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc
from src.proto.tensor_codec import tensor_to_proto

N_CTX = 1024


@dataclass(frozen=True)
class StepTiming:
    total_ms: float               # measured by the coordinator, around the whole step
    compute_ms: tuple[float, ...]  # measured inside each worker

    @property
    def overhead_ms(self) -> float:
        return self.total_ms - sum(self.compute_ms)   # gRPC + serialisation + Python


class Pipeline:
    def __init__(self, addresses: list[str], timeout_s: float = 10.0) -> None:
        self.addresses = addresses
        self.timeout_s = timeout_s
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

    def step(self, request_id: str, step_id: int, ids: list[int]) -> tuple[int, StepTiming]:
        """One decode step through every stage. Returns the next token id."""
        start = time.perf_counter()
        message = tensor_to_proto(torch.tensor([ids]))
        compute = []
        for stub in self.stubs:
            reply = stub.Forward(pb.ForwardRequest(request_id=request_id, step_id=step_id, input=message),
                                 timeout=self.timeout_s)
            if (reply.request_id, reply.step_id) != (request_id, step_id):
                raise RuntimeError(f"reply for {reply.request_id}/{reply.step_id}, expected {request_id}/{step_id}")
            compute.append(reply.compute_ms)
            message = reply.output                     # hidden state passes through untouched
        timing = StepTiming((time.perf_counter() - start) * 1000, tuple(compute))
        return reply.next_token, timing

    def close(self) -> None:
        for channel in self.channels:
            channel.close()


def generate(pipe: Pipeline, request_id: str, prompt_ids: list[int], max_new_tokens: int,
             eos_id: int | None = None) -> tuple[list[int], list[StepTiming]]:
    """Same contract as src.model.generate.generate, plus per-step timings."""
    ids = list(prompt_ids)
    new: list[int] = []
    timings: list[StepTiming] = []
    for step_id in range(max_new_tokens):
        if len(ids) >= N_CTX:
            break
        tok, timing = pipe.step(request_id, step_id, ids)
        ids.append(tok)
        new.append(tok)
        timings.append(timing)
        if tok == eos_id:
            break
    return new, timings
