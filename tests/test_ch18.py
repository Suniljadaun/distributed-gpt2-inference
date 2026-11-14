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


# ---------------- NumPy analysis and Matplotlib charts (src/bench/stats.py, plots.py) ----------------

def test_growth_fit_recovers_a_straight_line():
    from src.bench import stats
    fit = stats.growth([20 + 0.5 * i for i in range(1, 101)])
    assert fit["slope"] == pytest.approx(0.5) and fit["intercept"] == pytest.approx(20) and fit["r2"] == pytest.approx(1)


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    from src.bench import stats
    values = [47.2, 49.8, 46.8, 47.0, 48.2, 46.0]
    lo, hi = stats.bootstrap_ci(values)
    assert lo < sum(values) / len(values) < hi
    assert stats.bootstrap_ci(values) == (lo, hi)                   # fixed seed: same data, same interval


def test_gain_without_noise_has_a_zero_width_interval():
    from src.bench import stats
    g = stats.gain([10.0, 10.0, 10.0], [12.0, 12.0, 12.0])
    assert g["gain"] == pytest.approx(0.2) and g["ci"] == (pytest.approx(0.2), pytest.approx(0.2))


def _timeline(rates_static, rates_dynamic, a_ms):
    mk = lambda rates: [{"t": 4 * i, "tokens_per_s": r, "detail": f"A {a:4.1f} B 19.0 ms  0-2 | 3-5"}
                        for i, (r, a) in enumerate(zip(rates, a_ms))]
    return {"slow_at_s": 24, "timeline": {"static": mk(rates_static), "dynamic": mk(rates_dynamic)}}


def test_rebalance_analysis_finds_the_slowdown_from_the_data():
    from src.bench import stats
    data = _timeline([46, 48, 48, 48, 48, 30, 26, 26, 26], [44, 53, 53, 53, 53, 30, 38, 38, 38],
                     [10, 10, 10, 10, 10, 10.5, 35, 35, 35])
    an = stats.rebalance_analysis(data)
    assert an["slow_start_t"] == 24                                 # first bucket with A > 1.8x its usual time
    assert an["steady"]["n"] == (4, 4)                              # t = 4..16: skips warm-up and the t=20 change
    assert an["slowed"]["gain"] == pytest.approx(38 / 26 - 1)       # t >= 28


def test_matplotlib_writes_every_chart(tmp_path):
    from src.bench import plots
    data = {
        "kv_cache": {"step_ms": {"no cache": [20 + 0.4 * i for i in range(50)], "KV-cache": [22.0] * 50},
                     "tokens_per_s": {"no cache": 12.0, "KV-cache": 44.0}},
        "split_sweep": [{"split": "3/3", "A_ms": 8.0, "B_ms": 16.0, "overhead_ms": 3.0, "total_ms": 27.0}],
        "throughput": [{"config": "1 at a time", "tokens_per_s": 39.0}, {"config": "batch 24", "tokens_per_s": 142.0}],
        "load": [{"queue": "bounded 8", "rate": 1.0, "p50": 0.3, "p99": 0.5, "ok": 8, "rejected": 0, "expired": 0},
                 {"queue": "bounded 8", "rate": 6.0, "p50": 2.1, "p99": 2.7, "ok": 51, "rejected": 19, "expired": 0},
                 {"queue": "unbounded", "rate": 6.0, "p50": 2.5, "p99": 3.0, "ok": 33, "rejected": 0, "expired": 46}],
        "wire": [{"payload": "hidden, 1 token", "protobuf_ms": 0.03, "torch.save_ms": 0.5, "json_ms": 0.8}],
        "rebalance": _timeline([46, 48, 48, 48, 48, 30, 26, 26, 26], [44, 53, 53, 53, 53, 30, 38, 38, 38],
                               [10, 10, 10, 10, 10, 10.5, 35, 35, 35]),
    }
    for name, d in data.items():
        path = plots.draw(name, d, tmp_path)
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and path.stat().st_size > 10_000
