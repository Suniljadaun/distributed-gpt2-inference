"""
test_ch09.py - Chapter 9
The KV-cache gives the same tokens as no cache, and the workers keep each request's cache
separate, in order, and released.
Run:  python -m pytest tests/test_ch09.py -v
"""
import json

import grpc
import pytest
import torch

from src.coordinator.cluster import running_cluster, two_stage_layout
from src.model.config import GPTConfig
from src.model.generate import generate, generate_cached
from src.model.kv_cache import KVCache, LayerKV
from src.model.layers import CausalSelfAttention
from src.model.weights import load_pretrained
from src.proto import inference_pb2 as pb
from src.reference import REFERENCE_FILE

PROMPT = [464, 3139, 286, 4881, 318]
BYTES_PER_TOKEN = 2 * 6 * 768 * 4          # K and V, 6 layers, 768 numbers each, float32


@pytest.fixture(scope="module")
def model():
    return load_pretrained()


@pytest.fixture(scope="module")
def reference():
    return json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))


# ---------------- in one process ----------------

def test_layer_kv_grows_along_positions() -> None:
    kv = LayerKV()
    k, v = kv.append(torch.zeros(1, 12, 5, 64), torch.zeros(1, 12, 5, 64))
    assert kv.length == 5 and k.shape == (1, 12, 5, 64)
    k, v = kv.append(torch.ones(1, 12, 1, 64), torch.ones(1, 12, 1, 64))
    assert kv.length == 6 and k.shape == (1, 12, 6, 64) and v[0, 0, 5, 0] == 1
    assert kv.nbytes == 2 * 12 * 6 * 64 * 4


@torch.inference_mode()
def test_attention_with_cache_matches_without() -> None:
    """6 tokens at once == 4 tokens, then 1, then 1, reusing K and V."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(GPTConfig())
    x = torch.randn(1, 6, 768)
    full = attn(x)
    kv = LayerKV()
    parts = [attn(x[:, :4], kv), attn(x[:, 4:5], kv), attn(x[:, 5:6], kv)]
    assert torch.allclose(torch.cat(parts, dim=1), full, atol=1e-5)
    assert kv.length == 6


def test_cached_generation_matches_reference(model, reference) -> None:
    for entry in reference["entries"]:
        assert generate_cached(model, entry["prompt_ids"], reference["max_new_tokens"]) == entry["new_ids"]


def test_cached_equals_uncached_over_60_tokens(model) -> None:
    assert generate_cached(model, PROMPT, 60) == generate(model, PROMPT, 60)


@torch.inference_mode()
def test_cache_size_per_token(model) -> None:
    from src.model.generate import cached_step
    cache = KVCache()
    tok = cached_step(model, cache, PROMPT)            # prefill: 5 positions
    for _ in range(3):
        tok = cached_step(model, cache, [tok])          # +1 position each
    assert cache.length == 8
    assert cache.nbytes == 8 * BYTES_PER_TOKEN


# ---------------- across worker processes ----------------

@pytest.fixture(scope="module")
def pipe():
    with running_cluster(two_stage_layout(base_port=50101)) as p:
        yield p


def test_interleaved_requests_keep_separate_caches(pipe, reference) -> None:
    """Two requests advance one step at a time, alternating, on the same workers."""
    e1, e2 = reference["entries"][0], reference["entries"][1]
    state = {"x": [e1["prompt_ids"], 0, []], "y": [e2["prompt_ids"], 0, []]}   # feed, start_pos, out
    for step in range(reference["max_new_tokens"]):
        for rid in ("x", "y"):
            feed, start_pos, out = state[rid]
            tok, _ = pipe.step(rid, step, feed, start_pos)
            state[rid] = [[tok], start_pos + len(feed), out + [tok]]
    pipe.release("x")
    pipe.release("y")
    assert state["x"][2] == e1["new_ids"] and state["y"][2] == e2["new_ids"]


def test_step_out_of_order_is_rejected(pipe) -> None:
    pipe.step("gap", 0, PROMPT, 0)
    with pytest.raises(grpc.RpcError) as err:
        pipe.step("gap", 2, [262], 5)                   # step 1 never happened
    assert err.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    pipe.release("gap")


def test_unknown_request_is_rejected(pipe) -> None:
    with pytest.raises(grpc.RpcError) as err:
        pipe.step("never-started", 3, [262], 8)
    assert err.value.code() == grpc.StatusCode.FAILED_PRECONDITION


def test_wrong_start_pos_is_rejected(pipe) -> None:
    pipe.step("pos", 0, PROMPT, 0)
    with pytest.raises(grpc.RpcError) as err:
        pipe.step("pos", 1, [262], 4)                   # cache holds 5 positions, not 4
    assert err.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    pipe.release("pos")


def test_retry_of_last_step_gets_same_answer_without_growing_cache(pipe) -> None:
    tok0, _ = pipe.step("retry", 0, PROMPT, 0)
    tok1, _ = pipe.step("retry", 1, [tok0], 5)
    again, _ = pipe.step("retry", 1, [tok0], 5)         # e.g. the reply was lost and we resend
    assert again == tok1
    tok2, _ = pipe.step("retry", 2, [tok1], 6)          # and the request continues normally
    assert tok2 == 286                                  # 3rd reference token for this prompt
    pipe.release("retry")


def test_release_frees_the_cache(pipe) -> None:
    pipe.step("rel", 0, PROMPT, 0)
    before = pipe.stubs[0].Info(pb.InfoRequest(), timeout=5)
    reply = pipe.stubs[0].Release(pb.ReleaseRequest(request_id="rel"), timeout=5)
    after = pipe.stubs[0].Info(pb.InfoRequest(), timeout=5)
    assert reply.existed and reply.freed_bytes == 5 * BYTES_PER_TOKEN // 2    # worker A: 3 of 6 layers
    assert after.cached_requests == before.cached_requests - 1
    assert not pipe.stubs[0].Release(pb.ReleaseRequest(request_id="rel"), timeout=5).existed
    pipe.release("rel")                                 # worker B's copy
