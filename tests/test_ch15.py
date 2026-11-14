"""
test_ch15.py - Chapter 15
Layers move between running workers without changing a single output token: KV-cache slices
migrate with them, old-epoch calls are fenced, a failed preparation leaves the layout untouched,
a restarted worker comes back with its NEW layers, and the monitor moves layers off a slow worker.
Run:  python -m pytest tests/test_ch15.py -v
"""
import json
import threading
import time

import grpc
import pytest

from src.coordinator.cluster import WorkerSpec, running_cluster, supervised_cluster, two_stage_layout
from src.coordinator.pipeline import generate
from src.coordinator.rebalancer import Rebalancer
from src.proto import inference_pb2 as pb
from src.proto.tensor_codec import tensor_to_proto
from src.reference import REFERENCE_FILE

import torch

REFERENCE = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
ENTRIES = REFERENCE["entries"]
N = REFERENCE["max_new_tokens"]


def layout_of(pipe) -> list[tuple[int, int, int]]:
    return [(i.first_layer, i.last_layer, i.epoch) for i in pipe.check_layout(6)]


@pytest.fixture(scope="module")
def pipe():
    with running_cluster(two_stage_layout(base_port=50331)) as p:
        yield p


@pytest.fixture(scope="module")
def reb(pipe):
    return Rebalancer(pipe)


def test_move_a_layer_in_the_middle_of_a_request(pipe, reb) -> None:
    e = ENTRIES[0]
    tok, _ = pipe.step("m", 0, e["prompt_ids"], 0)
    out, pos = [tok], len(e["prompt_ids"])
    for s in range(1, 8):
        tok, _ = pipe.step("m", s, [tok], pos)
        out.append(tok)
        pos += 1
    report = reb.move(0, 1)                                         # layer 2: A -> B, cache slice included
    assert report.layer == 2 and report.kv_requests == 1 and report.kv_bytes > 0
    assert layout_of(pipe) == [(0, 2, 2), (2, 6, 2)]
    for s in range(8, N):
        tok, _ = pipe.step("m", s, [tok], pos)
        out.append(tok)
        pos += 1
    pipe.release("m")
    assert out == e["new_ids"]                                      # not one token changed


def test_move_back_and_forth_with_several_requests_in_flight(pipe, reb) -> None:
    rids = [f"multi-{i}" for i in range(len(ENTRIES))]
    toks = [pipe.step(rid, 0, e["prompt_ids"], 0)[0] for rid, e in zip(rids, ENTRIES)]
    positions = [len(e["prompt_ids"]) for e in ENTRIES]
    outs = [[t] for t in toks]
    moves = {5: (1, 0), 10: (1, 0), 15: (0, 1)}                     # at these steps, move a layer
    for step in range(1, N):
        if step in moves:
            reb.move(*moves[step])
        toks, _ = pipe.step_batch([(rid, step, t, p) for rid, t, p in zip(rids, toks, positions)])
        positions = [p + 1 for p in positions]
        for o, t in zip(outs, toks):
            o.append(t)
    for rid in rids:
        pipe.release(rid)
    assert outs == [e["new_ids"] for e in ENTRIES]
    assert layout_of(pipe)[0][:2] == (0, 3)                         # 2 -> 3 -> 4 -> 3 layers on A


def test_call_with_an_old_epoch_is_fenced(pipe, reb) -> None:
    stale = pb.ForwardRequest(request_id="old-epoch", step_id=0, input=tensor_to_proto(torch.tensor([[464, 3139]])),
                              expect_epoch=reb.epoch - 1)
    with pytest.raises(grpc.RpcError) as err:
        pipe.stubs[0].Forward(stale, timeout=5)
    assert err.value.code() == grpc.StatusCode.FAILED_PRECONDITION and "epoch" in err.value.details()


def test_failed_prepare_leaves_the_layout_untouched(pipe, reb) -> None:
    before = layout_of(pipe)
    with pytest.raises(grpc.RpcError):                              # layer 5 is not adjacent to A's range
        pipe.stubs[0].PrepareLayer(pb.PrepareLayerRequest(epoch=reb.epoch + 1, layer=5), timeout=10)
    reb._abort(pipe.stubs[0])
    assert layout_of(pipe) == before
    new_ids, _ = generate(pipe, "after-failed-prepare", ENTRIES[1]["prompt_ids"], N)
    assert new_ids == ENTRIES[1]["new_ids"]


def test_restarted_worker_comes_back_with_its_new_layers() -> None:
    with supervised_cluster(two_stage_layout(base_port=50333)) as (p, sup):
        reb = Rebalancer(p, supervisor=sup)
        reb.move(1, 0)                                              # layer 3: B -> A; A = 0..3, B = 4..5
        sup.kill(0)                                                 # A crashes after the move
        deadline = time.monotonic() + 30
        while not (sup.status()[0]["generation"] == 2 and sup.status()[0]["up"]):
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert layout_of(p) == [(0, 4, 2), (4, 6, 2)]               # same layers, same epoch as before the crash
        new_ids, _ = generate(p, "after-restart", ENTRIES[0]["prompt_ids"], N)
        assert new_ids == ENTRIES[0]["new_ids"]


def test_monitor_moves_layers_off_a_slow_worker() -> None:
    a, b = two_stage_layout(base_port=50335)
    slow_a = WorkerSpec(a.worker_id, a.port, a.first, a.last, embed=True, extra_args=("--block-delay-ms", "12"))
    with running_cluster([slow_a, b]) as p:
        reb = Rebalancer(p, period_s=0.5, window_s=2.0, cooldown_s=2.5, min_samples=10)
        reb.start()
        stop = time.monotonic() + 25

        def load() -> None:                                         # keep decode steps flowing
            k = 0
            while time.monotonic() < stop and len(reb.moves) < 2:
                generate(p, f"load-{k}", ENTRIES[k % 3]["prompt_ids"], 30)
                k += 1

        t = threading.Thread(target=load)
        t.start()
        t.join()
        reb.stop()
        assert len(reb.moves) >= 1 and all(m.src == 0 for m in reb.moves[:1])
        assert reb.layout[0][1] < 3                                 # slow A gave layers away
        new_ids, _ = generate(p, "after-auto", ENTRIES[2]["prompt_ids"], N)
        assert new_ids == ENTRIES[2]["new_ids"]
