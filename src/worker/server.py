"""
server.py - Chapter 8
A worker process: holds one Stage and serves it over gRPC.
Run:  python -m src.worker.server --worker-id A --port 50061 --first 0 --last 3 --embed
      python -m src.worker.server --worker-id B --port 50062 --first 3 --last 6 --head
"""
import argparse
import os
import time
from concurrent import futures

import grpc
import torch

from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc
from src.proto.tensor_codec import tensor_from_proto, tensor_to_proto
from src.worker.stage import Stage, load_stage


class StageWorkerService(pb_grpc.StageWorkerServicer):
    def __init__(self, worker_id: str, stage: Stage) -> None:
        self.worker_id = worker_id
        self.stage = stage

    def Info(self, request: pb.InfoRequest, context: grpc.ServicerContext) -> pb.InfoReply:
        s = self.stage
        return pb.InfoReply(
            worker_id=self.worker_id, first_layer=s.first, last_layer=s.last,
            has_embed=s.has_embed, has_head=s.has_head, pid=os.getpid(),
            param_bytes=sum(p.numel() * p.element_size() for p in s.parameters()),
        )

    def Forward(self, request: pb.ForwardRequest, context: grpc.ServicerContext) -> pb.ForwardReply:
        try:
            x = tensor_from_proto(request.input)
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"bad input tensor: {e}")

        start = time.perf_counter()
        try:
            with torch.inference_mode():
                out = self.stage(x)
        except ValueError as e:                                   # wrong shape/dtype for this stage
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        compute_ms = (time.perf_counter() - start) * 1000

        reply = pb.ForwardReply(request_id=request.request_id, step_id=request.step_id, compute_ms=compute_ms)
        if self.stage.has_head:
            reply.next_token = int(out[0, -1].argmax())            # greedy decoding happens here
        else:
            reply.output.CopyFrom(tensor_to_proto(out))
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
