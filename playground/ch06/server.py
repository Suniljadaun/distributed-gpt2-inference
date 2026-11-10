"""
server.py - Chapter 6
The Chapter 5 counter worker as a gRPC service: a thread pool, a lock, input validation,
deduplication by request_id, and a streaming method.
Run alone:  python -m playground.ch06.server --port 50556     (Ctrl+C to stop)
"""
import argparse
import os
import threading
import time
from concurrent import futures

import grpc

from playground.ch06 import counter_pb2 as pb
from playground.ch06 import counter_pb2_grpc as pb_grpc


class CounterService(pb_grpc.CounterServicer):
    def __init__(self) -> None:
        self.counter = 0
        self.done: dict[str, int] = {}     # request_id -> counter value we replied with
        self.lock = threading.Lock()       # several threads handle requests at once (Ch 1 race!)

    def Echo(self, request: pb.EchoRequest, context: grpc.ServicerContext) -> pb.EchoReply:
        return pb.EchoReply(payload=request.payload)

    def Add(self, request: pb.AddRequest, context: grpc.ServicerContext) -> pb.CounterReply:
        if not request.request_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request_id is required")
        if not 1 <= request.amount <= 1000:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"amount must be 1..1000, got {request.amount}")

        with self.lock:
            if request.request_id in self.done:                    # a retry of work already done
                print(f"    [server] duplicate {request.request_id!r}: not adding again", flush=True)
                return pb.CounterReply(counter=self.done[request.request_id])
            self.counter += request.amount
            self.done[request.request_id] = self.counter
            value = self.counter
        print(f"    [server] add {request.request_id!r} -> counter = {value}", flush=True)

        time.sleep(request.delay_s)                                # slow reply, as in Ch 5
        if not context.is_active():
            print(f"    [server] client stopped waiting for {request.request_id!r}; reply is dropped", flush=True)
        return pb.CounterReply(counter=value)

    def Get(self, request: pb.GetRequest, context: grpc.ServicerContext) -> pb.CounterReply:
        with self.lock:
            return pb.CounterReply(counter=self.counter)

    def CountUp(self, request: pb.CountUpRequest, context: grpc.ServicerContext):
        for i in range(request.n):
            if not context.is_active():                            # client cancelled: stop working
                return
            yield pb.Tick(i=i)                                     # each yield = one streamed message
            time.sleep(request.interval_s)


def serve(port: int) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_CounterServicer_to_server(CounterService(), server)
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    print(f"    [server pid={os.getpid()}] gRPC listening on 127.0.0.1:{port}", flush=True)
    server.wait_for_termination()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50556)
    serve(parser.parse_args().port)


if __name__ == "__main__":
    main()
