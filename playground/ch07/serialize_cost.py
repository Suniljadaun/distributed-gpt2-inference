"""
serialize_cost.py - Chapter 7, Measure it
Bytes on the wire and CPU time to encode + decode, for the tensors our engine will send,
using three formats: our protobuf Tensor, torch.save (pickle), and JSON text.
Run:  python -m playground.ch07.serialize_cost
"""
import io
import json
import time

import torch

from src.proto import inference_pb2 as pb
from src.proto.tensor_codec import tensor_from_proto, tensor_to_proto

PAYLOADS = [
    ("hidden, 1 token", (1, 1, 768)),
    ("hidden, 20-token prompt", (1, 20, 768)),
    ("hidden, 512-token prompt", (1, 512, 768)),
    ("KV, 1 layer, 512 tokens", (2, 1, 12, 512, 64)),
    ("KV, 6 layers, 512 tokens", (6, 2, 1, 12, 512, 64)),
]


def via_proto(t: torch.Tensor) -> tuple[int, torch.Tensor]:
    data = tensor_to_proto(t).SerializeToString()
    return len(data), tensor_from_proto(pb.Tensor.FromString(data))


def via_torch_save(t: torch.Tensor) -> tuple[int, torch.Tensor]:
    buf = io.BytesIO()
    torch.save(t, buf)
    data = buf.getvalue()
    return len(data), torch.load(io.BytesIO(data), weights_only=True)


def via_json(t: torch.Tensor) -> tuple[int, torch.Tensor]:
    data = json.dumps({"shape": list(t.shape), "data": t.reshape(-1).tolist()}).encode()
    obj = json.loads(data)
    return len(data), torch.tensor(obj["data"], dtype=torch.float32).reshape(obj["shape"])


def time_it(fn, t: torch.Tensor, runs: int) -> tuple[int, float, bool]:
    size, back = fn(t)                                 # warm-up, and check it is exact
    exact = torch.equal(back, t)
    start = time.perf_counter()
    for _ in range(runs):
        fn(t)
    return size, (time.perf_counter() - start) / runs, exact


def main() -> None:
    torch.manual_seed(0)
    print(f"{'payload':<26} {'raw bytes':>11} | {'format':<10} {'wire bytes':>11} {'enc+dec ms':>10} {'exact':>6}")
    for label, shape in PAYLOADS:
        t = torch.randn(shape)
        raw = t.nelement() * t.element_size()
        for name, fn in [("protobuf", via_proto), ("torch.save", via_torch_save), ("json", via_json)]:
            if name == "json" and raw > 2_000_000:
                print(f"{'':<26} {'':>11} | {name:<10} {'(skipped: too slow)':>30}")
                continue
            runs = 50 if raw < 1_000_000 else 5
            size, sec, exact = time_it(fn, t, runs)
            print(f"{label if name == 'protobuf' else '':<26} {f'{raw:,}' if name == 'protobuf' else '':>11} | "
                  f"{name:<10} {size:>11,} {sec * 1e3:>10.3f} {str(exact):>6}")
        print()


if __name__ == "__main__":
    main()
