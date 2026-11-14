"""
test_ch08.py - Chapter 8
Stages in one process first (fast, precise), then the real thing: two worker processes
reproducing results/reference.json.
Run:  python -m pytest tests/test_ch08.py -v
"""
import json

import grpc
import pytest
import torch

from src.coordinator.cluster import running_cluster, two_stage_layout
from src.coordinator.pipeline import generate
from src.model.weights import load_pretrained
from src.proto import inference_pb2 as pb
from src.proto.tensor_codec import tensor_to_proto
from src.reference import REFERENCE_FILE
from src.worker.stage import load_stage

BLOCK = 7_087_872
WTE, WPE, LN = 38_597_376, 786_432, 1_536


@pytest.fixture(scope="module")
def stages():
    return load_stage(0, 3, has_embed=True, has_head=False), load_stage(3, 6, has_embed=False, has_head=True)


def test_each_stage_holds_only_its_part(stages) -> None:
    a, b = stages
    assert sum(p.numel() for p in a.parameters()) == WTE + WPE + 3 * BLOCK
    assert sum(p.numel() for p in b.parameters()) == 3 * BLOCK + WTE + LN     # wte again: weight tying
    assert sorted(a.blocks.keys()) == ["0", "1", "2"] and sorted(b.blocks.keys()) == ["3", "4", "5"]


@torch.inference_mode()
def test_two_stages_equal_full_model(stages) -> None:
    a, b = stages
    ids = torch.tensor([[464, 3139, 286, 4881, 318]])
    full = load_pretrained()(ids)[:, -1:, :]
    split = b(a(ids))
    # Bit-identical on Windows. On Linux (CI) they differ by ~1e-5 on logits of ~70 (float32 rounding): stage B runs
    # the output layer on the last row only, the full model on all rows, and Linux's matmul kernels round
    # differently for different shapes. G1 is about TOKENS, so: equal to float32 rounding, same prediction.
    torch.testing.assert_close(split, full, rtol=0, atol=1e-4)
    assert split.argmax(-1).item() == full.argmax(-1).item()


def test_stage_rejects_wrong_input(stages) -> None:
    a, b = stages
    with pytest.raises(ValueError, match="expects hidden"):
        b(torch.tensor([[1, 2, 3]]))
    with pytest.raises(ValueError, match="expects token ids"):
        a(torch.zeros(1, 3, 768))


@pytest.fixture(scope="module")
def pipe():
    with running_cluster(two_stage_layout(base_port=50071)) as p:      # own ports: no clash with a demo
        yield p


def test_layout_is_checked(pipe) -> None:
    infos = pipe.check_layout(n_layer=6)
    assert [(i.worker_id, i.first_layer, i.last_layer) for i in infos] == [("A", 0, 3), ("B", 3, 6)]
    assert infos[0].pid != infos[1].pid


def test_distributed_matches_reference(pipe) -> None:
    """Goal G1 across processes."""
    reference = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
    for i, entry in enumerate(reference["entries"]):
        new_ids, _ = generate(pipe, f"test-{i}", entry["prompt_ids"], reference["max_new_tokens"])
        assert new_ids == entry["new_ids"], f"mismatch for {entry['prompt']!r}"


def test_worker_rejects_bad_input_with_invalid_argument(pipe) -> None:
    bad = pb.ForwardRequest(request_id="bad", step_id=0, input=tensor_to_proto(torch.zeros(1, 3, 768)))
    with pytest.raises(grpc.RpcError) as err:
        pipe.stubs[0].Forward(bad, timeout=5.0)                         # worker A wants token ids
    assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
