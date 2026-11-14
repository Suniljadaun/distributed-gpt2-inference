"""
client.py - send one prompt to a running coordinator and print the generated text.
    python -m src.client "The capital of France is" --tokens 30
    python -m src.client "Once upon a time" --address 127.0.0.1:50060 --request-id my-1
Works with the local coordinator (python -m src.coordinator.server) and with the Docker Compose one.
"""
import argparse
import sys
import time

import grpc

from src.model.tokenizer import Tokenizer
from src.proto import inference_pb2 as pb
from src.proto import inference_pb2_grpc as pb_grpc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--tokens", type=int, default=30, help="new tokens to generate (1..256)")
    parser.add_argument("--address", default="127.0.0.1:50060")
    parser.add_argument("--request-id", default="", help="reuse an id to see deduplication (default: a new one)")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    tok = Tokenizer()
    ids = tok.encode(args.prompt)
    rid = args.request_id or f"client-{time.time_ns()}"
    with grpc.insecure_channel(args.address) as channel:
        stub = pb_grpc.CoordinatorStub(channel)
        start = time.perf_counter()
        try:
            reply = stub.Generate(pb.GenerateRequest(request_id=rid, prompt_ids=ids, max_new_tokens=args.tokens),
                                  timeout=args.timeout)
        except grpc.RpcError as e:
            sys.exit(f"{e.code().name}: {e.details()}")
    took = time.perf_counter() - start
    print(args.prompt + tok.decode(list(reply.new_ids)))
    print(f"\n[{len(reply.new_ids)} tokens in {took:.2f} s, request id {rid}, replays {reply.replays}]",
          file=sys.stderr)


if __name__ == "__main__":
    main()
