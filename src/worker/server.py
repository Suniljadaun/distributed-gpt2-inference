"""
server.py - Chapter 8 (Chapter 9: per-request KV-cache, step ordering, Release)
A worker process: holds one Stage and serves it over gRPC.
Run:  python -m src.worker.server --worker-id A --port 50061 --first 0 --last 3 --embed
      python -m src.worker.server --worker-id B --port 50062 --first 3 --last 6 --head
"""
import argparse
import os
import threading
import time
from concurrent import futures

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


class StageWorkerService(pb_grpc.StageWorkerServicer):
    def __init__(self, worker_id: str, stage: Stage) -> None:
        self.worker_id = worker_id
        self.stage = stage
        self.requests: dict[str, RequestState] = {}
        self.requests_lock = threading.Lock()           # protects the dict itself

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
                return state.last_reply                     # a retry of the step we just did: same answer
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
            return reply


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--embed", action="store_true", help="this stage holds the embeddings")
    parser.add_argument("--head", action="store_true", help="this stage holds the output layer")
    args = parser.parse_args()

    stage = load_stage(args.first, args.last, args.embed, args.head)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_StageWorkerServicer_to_server(StageWorkerService(args.worker_id, stage), server)
    server.add_insecure_port(f"127.0.0.1:{args.port}")
    server.start()
    print(f"[worker {args.worker_id} pid={os.getpid()}] layers {args.first}..{args.last - 1}"
          f"{' +embed' if args.embed else ''}{' +head' if args.head else ''} on port {args.port}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
