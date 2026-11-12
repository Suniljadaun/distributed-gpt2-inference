"""
server.py (coordinator) - Chapter 10
The front door: a gRPC Coordinator service. Generate() admits a request into the Scheduler's
bounded queue (or rejects it), waits for the result, and turns the outcome into a status code.
Run (starts 2 workers + the coordinator):  python -m src.coordinator.server --port 50060
"""
import argparse
import time
from concurrent import futures

import grpc

from src.coordinator.cluster import running_cluster, two_stage_layout
from src.coordinator.pipeline import Pipeline
from src.coordinator.scheduler import Job, Scheduler
from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc

MAX_NEW_TOKENS = 256
DEFAULT_TIMEOUT_S = 30.0          # used when the caller sets no deadline


class CoordinatorService(pb_grpc.CoordinatorServicer):
    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler

    def Generate(self, request: pb.GenerateRequest, context: grpc.ServicerContext) -> pb.GenerateReply:
        if not request.prompt_ids:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "prompt_ids is empty")
        if not 1 <= request.max_new_tokens <= MAX_NEW_TOKENS:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"max_new_tokens must be 1..{MAX_NEW_TOKENS}")

        remaining = context.time_remaining()                    # the caller's deadline, in seconds
        if remaining is None:
            remaining = DEFAULT_TIMEOUT_S
        job = Job(request.request_id, list(request.prompt_ids), request.max_new_tokens,
                  deadline=time.monotonic() + remaining)
        if not self.scheduler.submit(job):
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                          f"queue full ({self.scheduler.queue_size} waiting); retry later")

        job.done.wait()                                         # the scheduler always finishes a job
        if job.error is not None:
            context.abort(*job.error)
        return pb.GenerateReply(
            request_id=job.request_id, new_ids=job.new_ids, retries=job.retries,
            queue_ms=(job.started - job.enqueued) * 1000,
            service_ms=(job.finished - job.started) * 1000,
        )

    def Stats(self, request: pb.StatsRequest, context: grpc.ServicerContext) -> pb.StatsReply:
        return pb.StatsReply(**self.scheduler.stats())


def start_coordinator(pipe: Pipeline, port: int, queue_size: int, slots: int = 1) -> tuple[grpc.Server, Scheduler]:
    scheduler = Scheduler(pipe, queue_size, slots)
    scheduler.start()
    # enough threads for every queued + running request to wait, plus a few for Stats/rejections
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=queue_size + slots + 8))
    pb_grpc.add_CoordinatorServicer_to_server(CoordinatorService(scheduler), server)
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    return server, scheduler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50060)
    parser.add_argument("--queue-size", type=int, default=8)
    args = parser.parse_args()
    with running_cluster(two_stage_layout(), timeout_s=2.0) as pipe:
        server, scheduler = start_coordinator(pipe, args.port, args.queue_size)
        print(f"[coordinator] listening on 127.0.0.1:{args.port}, queue size {args.queue_size}", flush=True)
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            pass
        finally:
            server.stop(grace=1.0)
            scheduler.stop()


if __name__ == "__main__":
    main()
