"""
timing.py - Chapter 2, demo 3
How fast is your CPU at the operations GPT-2 needs?
Run:  python playground/ch02/timing.py
"""
import time
from typing import Callable

import torch


def time_it(fn: Callable[[], object], runs: int = 20) -> float:
    """Average seconds per call, after one warm-up call."""
    fn()  # warm-up: the first call pays one-time setup costs
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - start) / runs


def main() -> None:
    torch.manual_seed(0)
    max_threads = torch.get_num_threads()
    print(f"torch {torch.__version__}, threads = {max_threads}")

    print("\n1. Square matmul: work grows as n^3")
    for n in [128, 256, 512, 1024, 2048]:
        a, b = torch.randn(n, n), torch.randn(n, n)
        s = time_it(lambda: a @ b, runs=5 if n >= 2048 else 20)
        print(f"  n={n:>5}  {s * 1e3:9.2f} ms  {2 * n**3 / s / 1e9:7.1f} GFLOP/s")

    print("\n2. One GPT-2 MLP matrix (768 x 3072): prefill vs decode")
    W = torch.randn(768, 3072)
    for seq in [1, 8, 64, 256]:
        x = torch.randn(seq, 768)
        s = time_it(lambda: x @ W, runs=50)
        flops = 2 * seq * 768 * 3072
        print(f"  tokens={seq:>4}  {s * 1e3:7.3f} ms  per token {s / seq * 1e6:8.1f} us"
              f"  {flops / s / 1e9:6.1f} GFLOP/s")

    print("\n3. Threads (1024 x 1024 matmul)")
    a, b = torch.randn(1024, 1024), torch.randn(1024, 1024)
    for t in sorted({1, 2, 4, max_threads}):
        torch.set_num_threads(t)
        s = time_it(lambda: a @ b, runs=10)
        print(f"  threads={t:>2}  {s * 1e3:7.2f} ms")
    torch.set_num_threads(max_threads)

    print("\n4. Autograd: what PyTorch remembers unless you tell it not to")
    lin = torch.nn.Linear(768, 3072)
    x = torch.randn(64, 768)
    y = lin(x)
    print(f"  normal          requires_grad={y.requires_grad}  grad_fn={type(y.grad_fn).__name__}")
    with torch.inference_mode():
        y = lin(x)
    print(f"  inference_mode  requires_grad={y.requires_grad}  grad_fn={y.grad_fn}")


if __name__ == "__main__":
    main()
