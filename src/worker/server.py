"""
server.py - Chapter 8 (Ch 9: KV-cache, step ordering, Release; Ch 10: fault injection; Ch 11: ForwardBatch)
A worker process: holds one Stage and serves it over gRPC.
Run:  python -m src.worker.server --worker-id A --port 50061 --first 0 --last 3 --embed
      python -m src.worker.server --worker-id B --port 50062 --first 3 --last 6 --head
"""
import argparse
import os
import threading
import time
from concurrent import futures
from contextlib import ExitStack
from dataclasses import dataclass

import grpc
import torch

from src.model.kv_cache import KVCache
from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc
from src.proto.tensor_codec import tensor_from_proto, tensor_to_proto
from src.worker.stage import Stage, load_stage


class RequestState:
    """Everything this worker remembers about one request between its steps."""

    def __init__(self) -> None:
        self.cache = KVCache()
        self.next_step = 0                              # the only step_id we will accept next
        self.last_reply: pb.ForwardReply | None = None  # to answer a retry of the previous step
        self.lock = threading.Lock()                    # one step of a request at a time


@dataclass(frozen=True)
class FaultPlan:
    """Ch 10: failures injected on purpose, to test retries and deadlines. All off by default."""
    fail_every: int = 0      # every Nth Forward call fails with UNAVAILABLE before doing anything
    slow_every: int = 0      # every Nth Forward call does its work, then replies late ...
    slow_ms: int = 0         # ... by this many milliseconds


class StageWorkerService(pb_grpc.StageWorkerServicer):
    def __init__(self, worker_id: str, stage: Stage, faults: FaultPlan = FaultPlan()) -> None:
        self.worker_id = worker_id
        self.stage = stage
        self.faults = faults
        self.requests: dict[str, RequestState] = {}
        self.requests_lock = threading.Lock()           # protects the dict itself
        self.calls = 0                                  # Forward calls so far, for fault injection
        self.calls_lock = threading.Lock()

    def Info(self, request: pb.InfoRequest, context: grpc.ServicerContext) -> pb.InfoReply:
        s = self.stage
        with self.requests_lock:
            states = list(self.requests.values())
        return pb.InfoReply(
            worker_id=self.worker_id, first_layer=s.first, last_layer=s.last,
            has_embed=s.has_embed, has_head=s.has_head, pid=os.getpid(),
            param_bytes=sum(p.numel() * p.element_size() for p in s.parameters()),
            cached_requests=len(states), kv_bytes=sum(st.cache.nbytes for st in states),
        )

    def Release(self, request: pb.ReleaseRequest, context: grpc.ServicerContext) -> pb.ReleaseReply:
        with self.requests_lock:
            state = self.requests.pop(request.request_id, None)
        if state is None:
            return pb.ReleaseReply(existed=False)
        return pb.ReleaseReply(existed=True, freed_bytes=state.cache.nbytes)

    def Forward(self, request: pb.ForwardRequest, context: grpc.ServicerContext) -> pb.ForwardReply:
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
                return state.last_reply, False              # a retry of the step we just did: same answer
            if request.step_id != state.next_step:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                              f"{request.request_id!r}: got step {request.step_id}, expected {state.next_step}")
            if request.start_pos != state.cache.length:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                              f"{request.request_id!r}: start_pos {request.start_pos} but cache holds {state.cache.length}")

            start = time.perf_counter()
            try:
                with torch.inference_mode():
                    out = self.stage(x, request.start_pos, state.cache)
            except ValueError as e:                         # wrong shape/dtype for this stage
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
            compute_ms = (time.perf_counter() - start) * 1000

            reply = pb.ForwardReply(request_id=request.request_id, step_id=request.step_id,
                                    compute_ms=compute_ms, cache_len=state.cache.length)
            if self.stage.has_head:
                reply.next_token = int(out[0, -1].argmax())  # greedy decoding happens here
            else:
                reply.output.CopyFrom(tensor_to_proto(out))
            state.next_step += 1
            state.last_reply = reply
            return reply, True

    def ForwardBatch(self, request: pb.ForwardBatchRequest, context: grpc.ServicerContext) -> pb.ForwardBatchReply:
        """Ch 11: one decode step (1 token) for several requests, computed as ONE batch."""
        items = list(request.items)
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
        compute_ms = 0.0
        with ExitStack() as locks:
            for i in sorted(range(len(items)), key=lambda i: ids[i]):   # fixed order: no deadlock
                locks.enter_context(states[i].lock)
            fresh = []                                                    # items to compute now
            for i, (it, st) in enumerate(zip(items, states)):
                if it.step_id == st.next_step - 1 and st.last_reply is not None:
                    replies[i] = st.last_reply                            # a retry: same answer
                elif it.step_id != st.next_step or it.start_pos != st.cache.length:
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                                  f"{it.request_id!r}: got step {it.step_id} at {it.start_pos}, "
                                  f"expected step {st.next_step} at {st.cache.length}")
                else:
                    fresh.append(i)
            if fresh:
                start = time.perf_counter()
                try:
                    with torch.inference_mode():
                        out = self.stage.forward_decode_batch(
                            torch.cat([xs[i] for i in fresh], dim=0),
                            [items[i].start_pos for i in fresh], [states[i].cache for i in fresh])
                except ValueError as e:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
                compute_ms = (time.perf_counter() - start) * 1000
                for row, i in enumerate(fresh):
                    st = states[i]
                    reply = pb.ForwardReply(request_id=ids[i], step_id=items[i].step_id,
                                            compute_ms=compute_ms, cache_len=st.cache.length)
                    if self.stage.has_head:
                        reply.next_token = int(out[row, -1].argmax())
                    else:
                        reply.output.CopyFrom(tensor_to_proto(out[row:row + 1]))
                    st.next_step += 1
                    st.last_reply = reply
                    replies[i] = reply
        return pb.ForwardBatchReply(items=replies, compute_ms=compute_ms)


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
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    stage = load_stage(args.first, args.last, args.embed, args.head)
    faults = FaultPlan(args.fail_every, args.slow_every, args.slow_ms)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_StageWorkerServicer_to_server(StageWorkerService(args.worker_id, stage, faults), server)
    server.add_insecure_port(f"127.0.0.1:{args.port}")
    server.start()
    print(f"[worker {args.worker_id} pid={os.getpid()}] layers {args.first}..{args.last - 1}"
          f"{' +embed' if args.embed else ''}{' +head' if args.head else ''} on port {args.port}"
          f"{f'  FAULTS {faults}' if faults != FaultPlan() else ''}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
