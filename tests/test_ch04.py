"""
test_ch04.py - Chapter 4
The full model vs Hugging Face, and the reference output (goal G1). Re-run this every chapter.
Run:  python -m pytest tests/test_ch04.py -v
"""
import json

import pytest
import torch
from transformers import GPT2LMHeadModel

from src.model.generate import generate
from src.model.tokenizer import Tokenizer
from src.model.weights import load_pretrained
from src.reference import REFERENCE_FILE


@pytest.fixture(scope="module")
def model():
    return load_pretrained()


@pytest.fixture(scope="module")
def tok():
    return Tokenizer()


@pytest.fixture(scope="module")
def hf():
    return GPT2LMHeadModel.from_pretrained("distilgpt2").eval()


def test_parameter_count(model) -> None:
    n = sum(p.numel() for p in model.parameters())
    assert n == 81_912_576          # 6 blocks + wte + wpe + ln_f; output layer shares wte


@torch.inference_mode()
def test_logits_match_hf(model, hf, tok) -> None:
    ids = torch.tensor([tok.encode("The capital of France is Paris, and the capital of Italy is")])
    ours = model(ids)
    theirs = hf(ids).logits
    assert ours.shape == theirs.shape == (1, ids.shape[1], 50257)
    assert torch.allclose(ours, theirs, atol=1e-3)
    assert torch.equal(ours.argmax(-1), theirs.argmax(-1))       # same predicted token everywhere


@torch.inference_mode()
def test_greedy_matches_hf_generate(model, hf, tok) -> None:
    prompt_ids = tok.encode("Distributed systems are hard because")
    ours = generate(model, prompt_ids, max_new_tokens=20)
    out = hf.generate(torch.tensor([prompt_ids]), max_new_tokens=20, do_sample=False,
                      pad_token_id=tok.eos_id)
    theirs = out[0, len(prompt_ids):].tolist()
    assert ours == theirs


def test_reference_output(model) -> None:
    """THE end-to-end correctness test of the project (goal G1)."""
    if not REFERENCE_FILE.exists():
        pytest.skip("run `python -m src.reference --save` first")
    data = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
    for entry in data["entries"]:
        new_ids = generate(model, entry["prompt_ids"], data["max_new_tokens"])
        assert new_ids == entry["new_ids"], f"mismatch for prompt {entry['prompt']!r}"
