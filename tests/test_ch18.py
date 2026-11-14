"""test_ch18.py - the live slowdown switch used by the demo (--block-delay-file)."""
import time

import torch

from src.coordinator.cluster import supervised_cluster, two_stage_layout
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
