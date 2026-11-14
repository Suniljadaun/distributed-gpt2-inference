"""test_ch18.py - the live slowdown switch used by the demo (--block-delay-file)."""
import time

import pytest
import torch

from src.coordinator.cluster import supervised_cluster, two_stage_layout
from src.coordinator.rebalancer import MoveReport, Rebalancer, StageModel
from src.worker.stage import load_stage


def _timed_forward(stage) -> float:
    x = torch.randn(1, 1, 768)
    start = time.perf_counter()
    with torch.no_grad():
        stage(x)
    return (time.perf_counter() - start) * 1000


def test_delay_applies_only_while_the_file_exists(tmp_path):
    stage = load_stage(3, 5, has_embed=False, has_head=False)      # 2 blocks
    stage.block_delay_ms = 30
    stage.delay_file = tmp_path / "slow.flag"
    fast = _timed_forward(stage)
    stage.delay_file.touch()
    slow = _timed_forward(stage)
    stage.delay_file.unlink()
    fast_again = _timed_forward(stage)
    assert slow >= 60                                               # 2 blocks x 30 ms
    assert fast < 40 and fast_again < 40


def test_wait_all_up_sees_a_crash_before_the_next_heartbeat():
    """The bug the demo rehearsal found: right after a crash, wait_all_up must not say 'all up'."""
    with supervised_cluster(two_stage_layout(base_port=50601)) as (pipe, sup):
        sup.kill(1)
        sup.workers[1].proc.wait()                                  # dead, but no heartbeat sweep yet
        assert sup.wait_all_up(0) is False                          # before the fix: True
        assert sup.wait_all_up(30) is True                          # the monitor restarts it
        assert sup.status()[1]["generation"] == 2


def test_no_file_means_always_delayed():
    stage = load_stage(3, 4, has_embed=False, has_head=False)
    stage.block_delay_ms = 30
    assert _timed_forward(stage) >= 30


def _policy(layout, last_move=None) -> Rebalancer:
    """A Rebalancer's decision logic only: no workers needed for best_move()."""
    reb = Rebalancer.__new__(Rebalancer)
    reb.layout, reb.margin, reb.undo_margin = list(layout), 0.05, 0.10
    reb.moves = [] if last_move is None else [MoveReport(3, *last_move, 2, 0, 0, 0, 0, tuple(layout))]
    return reb


def test_undoing_the_last_move_needs_a_bigger_gain():
    """Numbers from the flapping demo run: A 18.5 ms (5 blocks), B 14.3 ms (1 block + head).
    Moving layer 4 back to B predicts 18.5 -> 17.5 ms: 5.4%."""
    m = StageModel(stage_ms=(18.5, 14.3), per_block_ms=(3.6, 3.2), fixed_ms=(0.5, 11.1))
    fresh = _policy([(0, 5), (5, 6)])
    assert fresh.best_move(m) is not None                           # 5.4% > 5%: a fresh decision moves
    after_a_move_to_A = _policy([(0, 5), (5, 6)], last_move=(1, 0))
    assert after_a_move_to_A.best_move(m) is None                   # but it would UNDO the last move: < 10%


def test_a_large_gain_still_undoes_the_last_move():
    """After a slowdown ends, moving layers back is an undo, but gains are 13-17%: it must still happen."""
    m = StageModel(stage_ms=(3.8, 23.3), per_block_ms=(3.5, 3.0), fixed_ms=(0.3, 8.3))
    reb = _policy([(0, 1), (1, 6)], last_move=(0, 1))
    assert reb.best_move(m) == (pytest.approx(20.3, abs=0.1), 1, 0)
