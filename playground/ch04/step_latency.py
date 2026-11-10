"""
step_latency.py - Chapter 4, Measure it
Where does the time of one decode step go, and how does it grow without a KV-cache?
Run:  python -m playground.ch04.step_latency
"""
import time

import torch

from src.model.weights import load_pretrained


def time_it(fn, runs: int) -> float:
    fn()  # warm-up
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - start) / runs


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(0)
    model = load_pretrained()
    L = model.cfg.n_layer

    print("1. One decode step, split by stage (no KV-cache: the whole sequence is re-run)")
    print(f"{'seq':>5} {'embed ms':>9} {'6 blocks ms':>12} {'head(last) ms':>14} {'head(all) ms':>13} {'step ms':>8}")
    for s in [1, 8, 32, 128, 512]:
        ids = torch.randint(0, 50257, (1, s))
        x = model.embed(ids)
        h = model.run_blocks(x, 0, L)
        runs = 20 if s <= 128 else 5
        t_emb = time_it(lambda: model.embed(ids), runs)
        t_blocks = time_it(lambda: model.run_blocks(x, 0, L), runs)
        t_last = time_it(lambda: model.head(h[:, -1:, :]), runs)
        t_all = time_it(lambda: model.head(h), runs)
        print(f"{s:>5} {t_emb * 1e3:>9.2f} {t_blocks * 1e3:>12.2f} {t_last * 1e3:>14.2f} "
              f"{t_all * 1e3:>13.2f} {(t_emb + t_blocks + t_last) * 1e3:>8.2f}")

    print("\n2. Generating 200 tokens from a 5-token prompt: step time grows with length")
    ids = [464, 3139, 286, 4881, 318]
    marks = {1, 50, 100, 150, 200}
    total_start = time.perf_counter()
    for step in range(1, 201):
        t0 = time.perf_counter()
        h = model.run_blocks(model.embed(torch.tensor([ids])), 0, L)
        ids.append(int(model.head(h[:, -1:, :])[0, -1].argmax()))
        if step in marks:
            print(f"  step {step:>3}  seq={len(ids) - 1:>3}  {(time.perf_counter() - t0) * 1e3:6.1f} ms")
    total = time.perf_counter() - total_start
    print(f"  total {total:.2f} s  ->  {200 / total:.1f} tokens/s")


if __name__ == "__main__":
    main()
