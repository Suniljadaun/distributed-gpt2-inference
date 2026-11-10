"""
split_sweep.py - Chapter 8, Measure it
Where should the model be cut? Try every split point and see how the work divides.
Run:  python -m playground.ch08.split_sweep
"""
import statistics

from src.coordinator.cluster import running_cluster, two_stage_layout
from src.coordinator.pipeline import generate

PROMPT_IDS = [464, 3139, 286, 4881, 318]      # "The capital of France is"


def main() -> None:
    print(f"{'split':<14} {'A ms':>6} {'B ms':>6} {'overhead':>9} {'total':>7} {'slowest/fastest':>16}")
    for split in [1, 2, 3, 4, 5]:
        layout = two_stage_layout(base_port=50080 + 2 * split, split=split)
        with running_cluster(layout) as pipe:
            _, timings = generate(pipe, f"sweep-{split}", PROMPT_IDS, max_new_tokens=20)
        a = statistics.mean(t.compute_ms[0] for t in timings)
        b = statistics.mean(t.compute_ms[1] for t in timings)
        over = statistics.mean(t.overhead_ms for t in timings)
        total = statistics.mean(t.total_ms for t in timings)
        name = f"A 0..{split - 1} | B {split}..5"
        print(f"{name:<14} {a:>6.1f} {b:>6.1f} {over:>9.1f} {total:>7.1f} {max(a, b) / min(a, b):>15.2f}x")


if __name__ == "__main__":
    main()
