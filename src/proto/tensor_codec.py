"""
tensor_codec.py - Chapter 7
torch.Tensor <-> inference_pb2.Tensor (shape, dtype, raw bytes). Exact: no rounding, no text.
"""
import math
import sys

import torch

from src.proto import inference_pb2 as pb

assert sys.byteorder == "little", "the wire format is little-endian; this machine is not"

_TO_WIRE = {
    torch.float32: pb.FLOAT32,
    torch.float16: pb.FLOAT16,
    torch.bfloat16: pb.BFLOAT16,
    torch.int64: pb.INT64,
}
_FROM_WIRE = {wire: dtype for dtype, wire in _TO_WIRE.items()}


def tensor_to_proto(t: torch.Tensor) -> pb.Tensor:
    if t.dtype not in _TO_WIRE:
        raise ValueError(f"unsupported dtype {t.dtype}")
    t = t.detach().contiguous()                       # bytes must be in row-major order (Ch 2.2.3)
    data = t.reshape(-1).view(torch.uint8).numpy().tobytes()
    return pb.Tensor(shape=list(t.shape), dtype=_TO_WIRE[t.dtype], data=data)


def tensor_from_proto(msg: pb.Tensor) -> torch.Tensor:
    if msg.dtype not in _FROM_WIRE:
        raise ValueError(f"unknown wire dtype {msg.dtype}")
    dtype = _FROM_WIRE[msg.dtype]
    shape = tuple(msg.shape)
    expected = math.prod(shape) * torch.empty(0, dtype=dtype).element_size()
    if len(msg.data) != expected:
        raise ValueError(f"shape {shape} {dtype} needs {expected} bytes, got {len(msg.data)}")
    if expected == 0:
        return torch.empty(shape, dtype=dtype)
    flat = torch.frombuffer(bytearray(msg.data), dtype=dtype)   # bytearray: a writable copy we own
    return flat.reshape(shape)
