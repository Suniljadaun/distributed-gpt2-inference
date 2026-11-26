"""
server.py - Chapter 8 (Ch 9 KV-cache; Ch 10 faults; Ch 11 ForwardBatch; Ch 12 Ping; Ch 14 fencing; Ch 15 layer moves)
A worker process: holds one Stage and serves it over gRPC.
Run:  python -m src.worker.server --worker-id A --port 50061 --first 0 --last 3 --embed
      python -m src.worker.server --worker-id B --port 50062 --first 3 --last 6 --head
"""
import argparse
import math
import os
import threading
import time
import zlib
from concurrent import futures
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path

import grpc
import torch

from src.model.kv_cache import KVCache
from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc
from src.proto.tensor_codec import tensor_from_proto, tensor_to_proto
from src.model.layers import Block
from src.worker.stage import Stage, load_block, load_stage

# Ch 15: exporting a layer's KV-cache for many requests can exceed gRPC's 4 MB default (Ch 6)
BIG_MESSAGES = [("grpc.max_send_message_length", 256 << 20), ("grpc.max_receive_message_length", 256 << 20)]


def pin_to_cpus(cpus: list[int]) -> None:
    """Ch 15: run this process only on the given CPU cores, as if it had a machine of its own."""
    if hasattr(os, "sched_setaffinity"):                       # Linux
        os.sched_setaffinity(0, cpus)
        return
    import ctypes                                              # Windows
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    mask = sum(1 << c for c in cpus)
    if not kernel32.SetProcessAffinityMask(ctypes.c_void_p(kernel32.GetCurrentProcess()), ctypes.c_size_t(mask)):
        raise OSError(f"could not pin to CPUs {cpus}")


class RequestState:
    """Everything this worker remembers about one request between its steps."""

    def __init__(self) -> None:
        self.cache = KVCache()
        self.next_step = 0                              # the only step_id we will accept next
        self.last_reply: pb.ForwardReply | None = None  # to answer a retry of the previous step
        self.last_digest: int | None = None             # Ch 14: fingerprint of that step's input
        self.lock = threading.Lock()                    # one step of a request at a time


def step_digest(request: pb.ForwardRequest) -> int:
    """Ch 14: a fingerprint of everything that defines a step. A genuine retry has the same one."""
    return zlib.crc32(request.input.SerializeToString() + request.start_pos.to_bytes(4, "little"))


@dataclass(frozen=True)
class FaultPlan:
    """Ch 10: failures injected on purpose, to test retries and deadlines. All off by default."""
    fail_every: int = 0      # every Nth Forward call fails with UNAVAILABLE before doing anything
    slow_every: int = 0      # every Nth Forward call does its work, then replies late ...
    slow_ms: int = 0         # ... by this many milliseconds
    hang_after_s: float = 0  # Ch 12: after this many seconds, stop answering ANY call (alive but hung)


class StageWorkerService(pb_grpc.StageWorkerServicer):
    def __init__(self, worker_id: str, stage: Stage, faults: FaultPlan = FaultPlan(), generation: int = 1,
                 epoch: int = 1, model: str = "distilgpt2") -> None:
        self.worker_id = worker_id
        self.model = model                              # Ch 18: a received layer must come from the same weights
        self.stage = stage
        self.faults = faults
        self.generation = generation                    # Ch 12: which incarnation of this worker we are
        self.epoch = epoch                              # Ch 15: which layout we are serving
        self.compute_lock = nullcontext()               # Ch 15: threading.Lock() = one step at a time (--serial)
        self.pending: tuple[int, int, Block] | None = None          # Ch 15: (epoch, layer, block) being received
        self.pending_kv: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.started = time.monotonic()
        self.hang_at = self.started + faults.hang_after_s if faults.hang_after_s else math.inf
        self.requests: dict[str, RequestState] = {}
        self.requests_lock = threading.Lock()           # protects the dict itself
        self.calls = 0                                  # Forward calls so far, for fault injection
        self.calls_lock = threading.Lock()

    def _fence(self, request: pb.ForwardRequest, context: grpc.ServicerContext) -> None:
        """Ch 14: refuse a call meant for another incarnation of this worker."""
        if request.expect_generation and request.expect_generation != self.generation:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          f"fenced: {request.request_id!r} addressed generation {request.expect_generation}, "
                          f"this is generation {self.generation}")
        if request.expect_epoch and request.expect_epoch != self.epoch:                 # Ch 15
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          f"fenced: {request.request_id!r} assumed layout epoch {request.expect_epoch}, "
                          f"this worker is at epoch {self.epoch}")

    # ---------------- Ch 15: moving a layer, in two phases ----------------

    def PrepareLayer(self, request: pb.PrepareLayerRequest, context: grpc.ServicerContext) -> pb.PrepareLayerReply:
        """Phase 1 (no pause needed): load the weights of the layer we are about to receive."""
        self._maybe_hang()
        s, i = self.stage, request.layer
        if not ((i == s.last and not s.has_head) or (i == s.first - 1 and not s.has_embed)):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                          f"layer {i} is not adjacent to {s.first}..{s.last - 1} on a movable side")
        if request.epoch != self.epoch + 1:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"prepare for epoch {request.epoch}, I am at {self.epoch}")
        start = time.perf_counter()
        block = load_block(i, self.model)
        with self.requests_lock:
            self.pending, self.pending_kv = (request.epoch, i, block), {}
        return pb.PrepareLayerReply(load_ms=(time.perf_counter() - start) * 1000)

    def ExportKV(self, request: pb.ExportKVRequest, context: grpc.ServicerContext) -> pb.ExportKVReply:
        """Every in-flight request's K and V for one layer (the coordinator has paused all steps)."""
        self._maybe_hang()
        with self.requests_lock:
            items = list(self.requests.items())
        slices = []
        for rid, state in items:
            with state.lock:
                kv = state.cache.layers.get(request.layer)
                if kv is not None and kv.k is not None:
                    slices.append(pb.KVSlice(request_id=rid, k=tensor_to_proto(kv.k), v=tensor_to_proto(kv.v)))
        return pb.ExportKVReply(slices=slices)

    def ImportKV(self, request: pb.ImportKVRequest, context: grpc.ServicerContext) -> pb.ImportKVReply:
        self._maybe_hang()
        with self.requests_lock:
            if self.pending is None or self.pending[:2] != (request.epoch, request.layer):
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "ImportKV without a matching PrepareLayer")
            for sl in request.slices:
                self.pending_kv[sl.request_id] = (tensor_from_proto(sl.k), tensor_from_proto(sl.v))
        return pb.ImportKVReply(imported=len(request.slices))

    def CommitLayout(self, request: pb.CommitLayoutRequest, context: grpc.ServicerContext) -> pb.CommitLayoutReply:
        """Phase 2: switch to the new range. The receiver adds the prepared layer, the sender drops it."""
        self._maybe_hang()
        s = self.stage
        new_range, old_range = (request.first_layer, request.last_layer), (s.first, s.last)
        with self.requests_lock:
            if request.epoch == self.epoch and new_range == old_range:  # already there (a retried commit, or a
                return pb.CommitLayoutReply(epoch=self.epoch, first_layer=s.first, last_layer=s.last)  # restarted worker)
            if request.epoch != self.epoch + 1:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"commit for epoch {request.epoch}, I am at {self.epoch}")
            try:
                if new_range == old_range:
                    pass                                                   # this worker is not involved
                elif self.pending is not None and self.pending[0] == request.epoch:     # receiving a layer
                    _, layer, block = self.pending
                    s.add_block(layer, block)
                    for rid, state in self.requests.items():
                        if rid in self.pending_kv:
                            k, v = self.pending_kv[rid]
                            kv = state.cache.layer(layer)
                            kv.k, kv.v = k, v
                else:                                                      # giving a layer away
                    layer = s.last - 1 if new_range[1] == s.last - 1 else s.first
                    s.drop_block(layer)
                    for state in self.requests.values():
                        state.cache.layers.pop(layer, None)
            except ValueError as e:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
            if (s.first, s.last) != new_range:
                context.abort(grpc.StatusCode.INTERNAL, f"layout is {s.first}..{s.last}, expected {new_range}")
            self.epoch = request.epoch
            self.pending, self.pending_kv = None, {}
        return pb.CommitLayoutReply(epoch=self.epoch, first_layer=s.first, last_layer=s.last)

    def AbortLayout(self, request: pb.AbortLayoutRequest, context: grpc.ServicerContext) -> pb.AbortLayoutReply:
        with self.requests_lock:
            self.pending, self.pending_kv = None, {}
        return pb.AbortLayoutReply()

    def _maybe_hang(self) -> None:
        """Ch 12 fault: once hang_at has passed, every call blocks forever, like a deadlocked process."""
        if time.monotonic() >= self.hang_at:
            time.sleep(3600)

    def Ping(self, request: pb.PingRequest, context: grpc.ServicerContext) -> pb.PingReply:
        self._maybe_hang()
        with self.requests_lock:
            cached = len(self.requests)
        return pb.PingReply(worker_id=self.worker_id, generation=self.generation, cached_requests=cached,
                            uptime_s=time.monotonic() - self.started)

    def Info(self, request: pb.InfoRequest, context: grpc.ServicerContext) -> pb.InfoReply:
        self._maybe_hang()
        s = self.stage
        with self.requests_lock:
            states = list(self.requests.values())
        return pb.InfoReply(
            worker_id=self.worker_id, first_layer=s.first, last_layer=s.last,
            has_embed=s.has_embed, has_head=s.has_head, pid=os.getpid(),
            param_bytes=sum(p.numel() * p.element_size() for p in s.parameters()),
            cached_requests=len(states), kv_bytes=sum(st.cache.nbytes for st in states),
            generation=self.generation, epoch=self.epoch, model=self.model,
        )

    def Release(self, request: pb.ReleaseRequest, context: grpc.ServicerContext) -> pb.ReleaseReply:
        self._maybe_hang()
        with self.requests_lock:
            state = self.requests.pop(request.request_id, None)
        if state is None:
            return pb.ReleaseReply(existed=False)
        return pb.ReleaseReply(existed=True, freed_bytes=state.cache.nbytes)

    def Forward(self, request: pb.ForwardRequest, context: grpc.ServicerContext) -> pb.ForwardReply:
        self._maybe_hang()
        with self.calls_lock:
            self.calls += 1
            call_no = self.calls
        f = self.faults
        if f.fail_every and call_no % f.fail_every == 0:
            context.abort(grpc.StatusCode.UNAVAILABLE, f"injected failure on call {call_no}")
        reply, fresh = self._forward(request, context)
        if fresh and f.slow_every and call_no % f.slow_every == 0:
            time.sleep(f.slow_ms / 1000)                    # injected: the work is done, the reply is late
        return reply

    def _forward(self, request: pb.ForwardRequest, context: grpc.ServicerContext) -> tuple[pb.ForwardReply, bool]:
        """The Chapter 9 step logic. Returns (reply, fresh): fresh = computed now, not a stored retry answer."""
        self._fence(request, context)
        try:
            x = tensor_from_proto(request.input)
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"bad input tensor: {e}")

        with self.requests_lock:
            state = self.requests.get(request.request_id)
            if state is None:
                if request.step_id != 0:
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                                  f"no cache for {request.request_id!r}: step {request.step_id} needs step 0 first")
                state = self.requests[request.request_id] = RequestState()

        with state.lock:
            if request.step_id == state.next_step - 1 and state.last_reply is not None:
                if step_digest(request) != state.last_digest:   # Ch 14: same step number, different input
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                                  f"conflicting retry of {request.request_id!r} step {request.step_id}: "
                                  f"input differs from the original")
                return state.last_reply, False              # a retry of the step we just did: same answer
            if request.step_id != state.next_step:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                              f"{request.request_id!r}: got step {request.step_id}, expected {state.next_step}")
            if request.start_pos != state.cache.length:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                              f"{request.request_id!r}: start_pos {request.start_pos} but cache holds {state.cache.length}")

            with self.compute_lock:                         # Ch 15: with --serial, wait for our turn first
                start = time.perf_counter()
                try:
                    with torch.inference_mode():
                        out = self.stage(x, request.start_pos, state.cache)
                except ValueError as e:                     # wrong shape/dtype for this stage
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
                compute_ms = (time.perf_counter() - start) * 1000

            reply = pb.ForwardReply(request_id=request.request_id, step_id=request.step_id,
                                    compute_ms=compute_ms, cache_len=state.cache.length,
                                    generation=self.generation, blocks_ms=self.stage.last_blocks_ms)
            if self.stage.has_head:
                reply.next_token = int(out[0, -1].argmax())  # greedy decoding happens here
            else:
                reply.output.CopyFrom(tensor_to_proto(out))
            state.next_step += 1
            state.last_reply = reply
            state.last_digest = step_digest(request)
            return reply, True

    def ForwardBatch(self, request: pb.ForwardBatchRequest, context: grpc.ServicerContext) -> pb.ForwardBatchReply:
        """Ch 11: one decode step (1 token) for several requests, computed as ONE batch."""
        self._maybe_hang()
        items = list(request.items)
        for it in items:
            self._fence(it, context)
        ids = [it.request_id for it in items]
        if len(set(ids)) != len(ids):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "a request appears twice in one batch")
        try:
            xs = [tensor_from_proto(it.input) for it in items]
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"bad input tensor: {e}")
        with self.requests_lock:
            states = [self.requests.get(rid) for rid in ids]
        missing = [rid for rid, st in zip(ids, states) if st is None]
        if missing:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"no cache for {missing}: prefill with Forward first")

        replies: list[pb.ForwardReply | None] = [None] * len(items)
        compute_ms = blocks_ms = 0.0
        with ExitStack() as locks:
            for i in sorted(range(len(items)), key=lambda i: ids[i]):   # fixed order: no deadlock
                locks.enter_context(states[i].lock)
            fresh = []                                                    # items to compute now
            for i, (it, st) in enumerate(zip(items, states)):
                if it.step_id == st.next_step - 1 and st.last_reply is not None:
                    if step_digest(it) != st.last_digest:                 # Ch 14
                        context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                                      f"conflicting retry of {it.request_id!r} step {it.step_id}")
                    replies[i] = st.last_reply                            # a retry: same answer
                elif it.step_id != st.next_step or it.start_pos != st.cache.length:
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                                  f"{it.request_id!r}: got step {it.step_id} at {it.start_pos}, "
                                  f"expected step {st.next_step} at {st.cache.length}")
                else:
                    fresh.append(i)
            if fresh:
                with self.compute_lock:
                    start = time.perf_counter()
                    try:
                        with torch.inference_mode():
                            out = self.stage.forward_decode_batch(
                                torch.cat([xs[i] for i in fresh], dim=0),
                                [items[i].start_pos for i in fresh], [states[i].cache for i in fresh])
                    except ValueError as e:
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
                    compute_ms = (time.perf_counter() - start) * 1000
                    blocks_ms = self.stage.last_blocks_ms
                for row, i in enumerate(fresh):
                    st = states[i]
                    reply = pb.ForwardReply(request_id=ids[i], step_id=items[i].step_id,
                                            compute_ms=compute_ms, cache_len=st.cache.length,
                                            generation=self.generation)
                    if self.stage.has_head:
                        reply.next_token = int(out[row, -1].argmax())
                    else:
                        reply.output.CopyFrom(tensor_to_proto(out[row:row + 1]))
                    st.next_step += 1
                    st.last_reply = reply
                    st.last_digest = step_digest(items[i])
                    replies[i] = reply
        return pb.ForwardBatchReply(items=replies, compute_ms=compute_ms, blocks_ms=blocks_ms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--embed", action="store_true", help="this stage holds the embeddings")
    parser.add_argument("--head", action="store_true", help="this stage holds the output layer")
    parser.add_argument("--fail-every", type=int, default=0, help="Ch 10 fault injection")
    parser.add_argument("--slow-every", type=int, default=0, help="Ch 10 fault injection")
    parser.add_argument("--slow-ms", type=int, default=0, help="Ch 10 fault injection")
    parser.add_argument("--threads", type=int, default=0, help="Ch 11: torch threads (0 = PyTorch's default)")
    parser.add_argument("--generation", type=int, default=1,
                        help="Ch 12: incarnation number, set by the supervisor. Ch 17: 0 = auto, the start time in "
                             "ms (for containers restarted by Docker, which reruns the same command)")
    parser.add_argument("--host", default="127.0.0.1", help="Ch 17: address to listen on (0.0.0.0 in a container)")
    parser.add_argument("--hang-after-s", type=float, default=0, help="Ch 12 fault injection")
    parser.add_argument("--epoch", type=int, default=1, help="Ch 15: layout epoch, set by the supervisor on restart")
    parser.add_argument("--block-delay-ms", type=float, default=0, help="Ch 15 fault: every block takes this much longer")
    parser.add_argument("--block-delay-after-s", type=float, default=0, help="Ch 15 fault: ... starting after this long")
    parser.add_argument("--block-delay-file", default="", help="Ch 18: ... and only while this file exists")
    parser.add_argument("--model", default="distilgpt2", help="Ch 18: distilgpt2 (6 layers) or gpt2 (12 layers)")
    parser.add_argument("--cpus", default="", help="Ch 15: pin to these cores, e.g. 0,1,2 (a 'machine' of its own)")
    parser.add_argument("--serial", action="store_true", help="Ch 15: compute one step at a time (a pipeline stage)")
    args = parser.parse_args()
    if args.generation == 0:                    # Ch 17: every (re)start gets a new, larger generation
        args.generation = time.time_ns() // 1_000_000

    if args.cpus:
        pin_to_cpus([int(c) for c in args.cpus.split(",")])
    if args.threads:
        torch.set_num_threads(args.threads)
    stage = load_stage(args.first, args.last, args.embed, args.head, args.model)
    stage.block_delay_ms, stage.slow_from = args.block_delay_ms, time.monotonic() + args.block_delay_after_s
    stage.delay_file = Path(args.block_delay_file) if args.block_delay_file else None
    faults = FaultPlan(args.fail_every, args.slow_every, args.slow_ms, args.hang_after_s)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8), options=BIG_MESSAGES)
    service = StageWorkerService(args.worker_id, stage, faults, args.generation, args.epoch, args.model)
    if args.serial:
        service.compute_lock = threading.Lock()
    pb_grpc.add_StageWorkerServicer_to_server(service, server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    print(f"[worker {args.worker_id} gen {args.generation} pid={os.getpid()}] {args.model} layers {args.first}..{args.last - 1}"
          f"{' +embed' if args.embed else ''}{' +head' if args.head else ''} on port {args.port}"
          f"{f'  FAULTS {faults}' if faults != FaultPlan() else ''}"
          f"{f'  +{args.block_delay_ms} ms per block after {args.block_delay_after_s} s' if args.block_delay_ms else ''}",
          flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
