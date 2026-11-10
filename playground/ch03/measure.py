"""
measure.py - Chapter 3, Measure it
How attention and the MLP scale with sequence length.
Run:  python -m playground.ch03.measure
"""
import time

import torch

from src.model.config import GPTConfig
from src.model.layers import MLP, CausalSelfAttention


def time_it(fn, runs: int) -> float:
    fn()  # warm-up
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - start) / runs


def main() -> None:
    torch.manual_seed(0)
    cfg = GPTConfig()
    attn, mlp = CausalSelfAttention(cfg), MLP(cfg)
    print(f"{'seq':>5} {'attention ms':>13} {'mlp ms':>8} {'attn/mlp':>9} {'score table MB':>15}")
    with torch.inference_mode():
        for s in [16, 64, 256, 512, 1024]:
            x = torch.randn(1, s, cfg.d_model)
            runs = 20 if s <= 256 else 5
            t_attn = time_it(lambda: attn(x), runs)
            t_mlp = time_it(lambda: mlp(x), runs)
            score_mb = cfg.n_head * s * s * 4 / 1e6      # [1, h, s, s] float32
            print(f"{s:>5} {t_attn * 1e3:>13.2f} {t_mlp * 1e3:>8.2f} {t_attn / t_mlp:>9.2f} {score_mb:>15.1f}")


if __name__ == "__main__":
    main()
