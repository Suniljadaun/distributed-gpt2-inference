"""
reference.py - Chapter 4
Generate text for fixed prompts with the single-process model and save it as the REFERENCE OUTPUT.
Every later chapter (distributed, cached, batched, crashed, rebalanced) must reproduce it exactly.
Run:  python -m src.reference            (print only)
      python -m src.reference --save     (also write results/reference.json)
"""
import argparse
import json
import time
from pathlib import Path

import torch

from src.model.generate import generate
from src.model.tokenizer import Tokenizer
from src.model.weights import load_pretrained

PROMPTS = [
    "The capital of France is",
    "Distributed systems are hard because",
    "Once upon a time, there was a",
]
MAX_NEW_TOKENS = 20
REFERENCE_FILE = Path(__file__).resolve().parent.parent / "results" / "reference.json"


def reference_file(model: str = "distilgpt2") -> Path:
    """Ch 18: results/reference.json for distilgpt2, results/reference_gpt2.json for gpt2."""
    return REFERENCE_FILE if model == "distilgpt2" else REFERENCE_FILE.with_name(f"reference_{model}.json")


def build_reference(model_name: str = "distilgpt2") -> list[dict]:
    tok = Tokenizer()
    model = load_pretrained(model_name)
    entries = []
    for prompt in PROMPTS:
        prompt_ids = tok.encode(prompt)
        start = time.perf_counter()
        new_ids = generate(model, prompt_ids, MAX_NEW_TOKENS, eos_id=tok.eos_id)
        seconds = time.perf_counter() - start
        entries.append({
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "new_ids": new_ids,
            "text": tok.decode(new_ids),
            "ms_per_token": round(seconds * 1000 / len(new_ids), 1),
        })
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="write results/reference.json")
    parser.add_argument("--model", default="distilgpt2", help="Ch 18: or gpt2 (saved as results/reference_gpt2.json)")
    args = parser.parse_args()

    entries = build_reference(args.model)
    for e in entries:
        print(f"{e['prompt']!r}")
        print(f"  -> {e['text']!r}")
        print(f"     {len(e['new_ids'])} tokens, {e['ms_per_token']} ms/token")

    if args.save:
        path = reference_file(args.model)
        path.parent.mkdir(exist_ok=True)
        data = {"model": args.model, "torch": torch.__version__, "decoding": "greedy",
                "max_new_tokens": MAX_NEW_TOKENS, "entries": entries}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
