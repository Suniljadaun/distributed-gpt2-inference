"""
server.py (coordinator) - Chapter 10 (Ch 11 batching; Ch 12 supervised workers; Ch 13 token log)
The front door: a gRPC Coordinator service. Generate() admits a request into the Scheduler's
bounded queue (or rejects it), waits for the result, and turns the outcome into a status code.
Run (starts 2 workers + the coordinator):  python -m src.coordinator.server --port 50060
"""
import argparse
import time
from concurrent import futures

import grpc

from src.coordinator.cluster import supervised_cluster, two_stage_layout
from src.coordinator.pipeline import Pipeline
from src.coordinator.scheduler import BatchingScheduler, Job, Scheduler
from src.coordinator.token_log import TokenLog
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

        log = self.scheduler.log                                # Ch 13: already answered? reply from the log
        if log is not None and log.has(request.request_id):
            rec = log.get(request.request_id)
            if rec.status == "completed":
                return pb.GenerateReply(request_id=request.request_id, new_ids=rec.tokens)

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
            request_id=job.request_id, new_ids=job.new_ids, retries=job.retries, replays=job.replays,
            queue_ms=(job.started - job.enqueued) * 1000,
            service_ms=(job.finished - job.started) * 1000,
        )

    def Stats(self, request: pb.StatsRequest, context: grpc.ServicerContext) -> pb.StatsReply:
        return pb.StatsReply(**self.scheduler.stats())


def start_coordinator(pipe: Pipeline, port: int, queue_size: int, slots: int = 1,
                      max_batch: int = 0, health=None, log: TokenLog | None = None) -> tuple[grpc.Server, Scheduler]:
    """slots = requests (or, with batching, micro-batch groups) in flight; max_batch > 0 turns on Ch 11 batching.
    health = a Ch 12 Supervisor: new work waits while a worker is being restarted.
    log = a Ch 13 TokenLog: tokens are committed to it and requests are replayed after worker failures."""
    if max_batch > 0:
        if log is not None:
            raise ValueError("the token log is supported with the plain scheduler only (Ch 13 Exercise 4)")
        scheduler = BatchingScheduler(pipe, queue_size, max_batch=max_batch, groups=slots, health=health)
        in_flight = slots * max_batch
    else:
        scheduler = Scheduler(pipe, queue_size, slots, health=health, log=log)
        in_flight = slots
    scheduler.start()
    # enough threads for every queued + running request to wait, plus a few for Stats/rejections
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=queue_size + in_flight + 8))
    pb_grpc.add_CoordinatorServicer_to_server(CoordinatorService(scheduler), server)
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    return server, scheduler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50060)
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--slots", type=int, default=1, help="Ch 11: requests / micro-batch groups in flight")
    parser.add_argument("--max-batch", type=int, default=0, help="Ch 11: >0 turns on continuous batching")
    parser.add_argument("--log", default="", help="Ch 13: token log file, e.g. logs/token_log.jsonl")
    args = parser.parse_args()
    log = TokenLog(args.log) if args.log else None
    if log is not None and log.incomplete():
        print(f"[coordinator] log has {len(log.incomplete())} unfinished requests (resumable)", flush=True)
    with supervised_cluster(two_stage_layout()) as (pipe, supervisor):
        server, scheduler = start_coordinator(pipe, args.port, args.queue_size, args.slots, args.max_batch,
                                              health=supervisor, log=log)
        print(f"[coordinator] listening on 127.0.0.1:{args.port}, queue {args.queue_size}, "
              f"slots {args.slots}, max batch {args.max_batch}, log {args.log or 'off'}", flush=True)
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            pass
        finally:
            server.stop(grace=1.0)
            scheduler.stop()


if __name__ == "__main__":
    main()
