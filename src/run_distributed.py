"""
run_distributed.py - Chapter 8
Start two workers, generate the reference prompts through them, and compare with
results/reference.json token for token. Prints where each step's time went.
Run:  python -m src.run_distributed
"""
import json
import statistics

from src.coordinator.cluster import running_cluster, two_stage_layout
from src.coordinator.pipeline import generate
from src.model.tokenizer import Tokenizer
from src.reference import REFERENCE_FILE


def main() -> None:
    tok = Tokenizer()
    reference = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))

    with running_cluster(two_stage_layout()) as pipe:
        for info in pipe.check_layout(n_layer=6):
            parts = [f"layers {info.first_layer}..{info.last_layer - 1}"]
            parts += ["embed"] if info.has_embed else []
            parts += ["head"] if info.has_head else []
            print(f"worker {info.worker_id} pid={info.pid:<6} {' + '.join(parts):<24} {info.param_bytes / 1e6:6.1f} MB")

        all_match = True
        for i, entry in enumerate(reference["entries"]):
            new_ids, timings = generate(pipe, f"req-{i}", entry["prompt_ids"], reference["max_new_tokens"])
            match = new_ids == entry["new_ids"]
            all_match &= match
            print(f"\n{entry['prompt']!r}")
            print(f"  -> {tok.decode(new_ids)!r}")
            print(f"  {'MATCHES reference' if match else 'DIFFERENT from reference'}")
            print(f"  per step: total {statistics.mean(t.total_ms for t in timings):5.1f} ms"
                  f" = A {statistics.mean(t.compute_ms[0] for t in timings):5.1f}"
                  f" + B {statistics.mean(t.compute_ms[1] for t in timings):5.1f}"
                  f" + overhead {statistics.mean(t.overhead_ms for t in timings):4.1f}")

    print("\nALL PROMPTS MATCH THE REFERENCE" if all_match else "\nMISMATCH: see above")


if __name__ == "__main__":
    main()
