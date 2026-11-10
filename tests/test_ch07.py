"""
test_ch07.py - Chapter 7
Tensors survive the trip to bytes and back EXACTLY, and a model split across that trip
gives bit-identical logits.
Run:  python -m pytest tests/test_ch07.py -v
"""
import pytest
import torch

from src.proto import inference_pb2 as pb
from src.proto.tensor_codec import tensor_from_proto, tensor_to_proto
from src.model.weights import load_pretrained


def wire_round_trip(t: torch.Tensor) -> torch.Tensor:
    """Tensor -> message -> bytes (what gRPC sends) -> message -> tensor."""
    data = tensor_to_proto(t).SerializeToString()
    return tensor_from_proto(pb.Tensor.FromString(data))


@pytest.mark.parametrize("shape, dtype", [
    ((1, 1, 768), torch.float32),       # one token's hidden state
    ((1, 20, 768), torch.float32),      # a prompt
    ((1, 12, 7, 64), torch.float32),    # a K or V slice (Ch 9)
    ((1, 1, 768), torch.float16),
    ((1, 1, 768), torch.bfloat16),
    ((), torch.float32),                # a scalar: shape is empty
    ((0, 768), torch.float32),          # empty tensor
], ids=["token", "prompt", "kv_slice", "fp16", "bf16", "scalar", "empty"])
def test_round_trip_exact(shape, dtype) -> None:
    t = torch.randn(shape).to(dtype)
    back = wire_round_trip(t)
    assert back.shape == t.shape and back.dtype == t.dtype
    assert torch.equal(back, t)                      # bit-exact, not allclose


def test_token_ids_round_trip() -> None:
    ids = torch.tensor([[464, 3139, 286, 4881, 318]])
    assert torch.equal(wire_round_trip(ids), ids)


def test_non_contiguous_input() -> None:
    t = torch.arange(12, dtype=torch.float32).view(3, 4).T    # transposed: not contiguous
    assert not t.is_contiguous()
    assert torch.equal(wire_round_trip(t), t)


def test_message_size_is_raw_bytes_plus_tiny_header() -> None:
    size = len(tensor_to_proto(torch.zeros(1, 1, 768)).SerializeToString())
    assert 3072 < size <= 3072 + 16


def test_corrupt_message_is_rejected() -> None:
    msg = tensor_to_proto(torch.zeros(1, 1, 768))
    msg.data = msg.data[:-4]                          # one float missing
    with pytest.raises(ValueError, match="needs 3072 bytes"):
        tensor_from_proto(msg)


def test_unsupported_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported dtype"):
        tensor_to_proto(torch.zeros(3, dtype=torch.int32))


@torch.inference_mode()
def test_split_model_across_the_wire_is_bit_identical() -> None:
    """Preview of Chapter 8: layers 0-2, serialise, deserialise, layers 3-5. Same logits, bit for bit."""
    model = load_pretrained()
    ids = torch.tensor([[464, 3139, 286, 4881, 318]])
    full = model(ids)

    h = model.run_blocks(model.embed(ids), 0, 3)              # "worker A"
    h = wire_round_trip(h)                                    # the network
    split = model.head(model.run_blocks(h, 3, 6))             # "worker B"
    assert torch.equal(split, full)
