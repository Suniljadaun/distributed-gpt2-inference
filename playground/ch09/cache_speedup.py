"""
cache_speedup.py - Chapter 9, Measure it
200 tokens with and without the KV-cache, in one process: step time as the text grows,
tokens per second, and how much memory the cache takes.
Run:  python -m playground.ch09.cache_speedup
"""
import time

import torch

from src.model.generate import cached_step, next_token
from src.model.kv_cache import KVCache
from src.model.weights import load_pretrained

PROMPT = [464, 3139, 286, 4881, 318]      # "The capital of France is"
STEPS = 200
MARKS = {1, 50, 100, 150, 200}


def run(model, use_cache: bool) -> tuple[dict[int, float], float, KVCache | None]:
    ids, cache, feed = list(PROMPT), KVCache(), list(PROMPT)
    step_ms: dict[int, float] = {}
    start = time.perf_counter()
    for step in range(1, STEPS + 1):
        t0 = time.perf_counter()
        tok = cached_step(model, cache, feed) if use_cache else next_token(model, ids)
        step_ms[step] = (time.perf_counter() - t0) * 1000
        ids.append(tok)
        feed = [tok]
    return step_ms, time.perf_counter() - start, cache if use_cache else None


def main() -> None:
    torch.manual_seed(0)
    model = load_pretrained()
    plain_ms, plain_s, _ = run(model, use_cache=False)
    cached_ms, cached_s, cache = run(model, use_cache=True)

    print(f"{'step':>5} {'seq':>5} {'no cache ms':>12} {'cache ms':>9} {'speed-up':>9} {'KV-cache MB':>12}")
    per_token_kv = 2 * model.cfg.n_layer * model.cfg.d_model * 4      # K and V, all layers, float32
    for step in sorted(MARKS):
        seq = len(PROMPT) + step - 1                                   # tokens the model has seen
        kv_mb = seq * per_token_kv / 1e6                               # all of them are cached
        print(f"{step:>5} {seq:>5} {plain_ms[step]:>12.1f} {cached_ms[step]:>9.1f} "
              f"{plain_ms[step] / cached_ms[step]:>8.1f}x {kv_mb:>12.2f}")
    print(f"\n200 tokens: no cache {plain_s:.2f} s ({STEPS / plain_s:.1f} tok/s)   "
          f"cache {cached_s:.2f} s ({STEPS / cached_s:.1f} tok/s)   speed-up {plain_s / cached_s:.1f}x")
    print(f"KV-cache after 200 tokens: {cache.length} positions, {cache.nbytes / 1e6:.2f} MB "
          f"({per_token_kv:,} bytes per token)")


if __name__ == "__main__":
    main()
