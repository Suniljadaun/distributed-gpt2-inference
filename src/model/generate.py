"""
generate.py - Chapter 4
Greedy text generation: repeatedly run the model and append the most likely next token.
No KV-cache yet: every step re-runs the WHOLE sequence (Chapter 9 fixes that).
"""
import torch

from src.model.gpt import GPT
from src.model.kv_cache import KVCache


@torch.inference_mode()
def next_token(model: GPT, ids: list[int]) -> int:
    """One decode step: the id of the most likely token to follow `ids`."""
    x = model.embed(torch.tensor([ids]))
    x = model.run_blocks(x, 0, model.cfg.n_layer)
    logits = model.head(x[:, -1:, :])          # only the last position predicts the next token
    return int(logits[0, -1].argmax())


def generate(model: GPT, prompt_ids: list[int], max_new_tokens: int, eos_id: int | None = None) -> list[int]:
    """Return ONLY the new token ids (not the prompt). Stops early at eos_id."""
    ids = list(prompt_ids)                      # copy: never modify the caller's list
    new: list[int] = []
    for _ in range(max_new_tokens):
        if len(ids) >= model.cfg.n_ctx:         # no position embedding beyond 1024
            break
        tok = next_token(model, ids)
        ids.append(tok)
        new.append(tok)
        if tok == eos_id:
            break
    return new


@torch.inference_mode()
def cached_step(model: GPT, cache: KVCache, new_ids: list[int]) -> int:
    """Run only the NEW tokens, at positions cache.length.., and extend the cache."""
    x = model.embed(torch.tensor([new_ids]), start_pos=cache.length)
    x = model.run_blocks(x, 0, model.cfg.n_layer, cache)
    return int(model.head(x[:, -1:, :])[0, -1].argmax())


def generate_cached(model: GPT, prompt_ids: list[int], max_new_tokens: int,
                    eos_id: int | None = None) -> list[int]:
    """Same contract as generate(), but with a KV-cache: step 0 = the prompt, then 1 token per step."""
    cache = KVCache()
    new: list[int] = []
    feed = list(prompt_ids)                     # step 0 feeds the whole prompt (prefill)
    for _ in range(max_new_tokens):
        if cache.length + len(feed) > model.cfg.n_ctx:
            break
        tok = cached_step(model, cache, feed)
        new.append(tok)
        feed = [tok]                            # every later step feeds just the newest token
        if tok == eos_id:
            break
    return new

