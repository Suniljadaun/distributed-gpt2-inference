"""
peek.py - Chapter 3, demo
Look inside: how text becomes tokens, and the shape of the data at every step of one Block.
Run:  python -m playground.ch03.peek
"""
import torch

from src.model.config import GPTConfig
from src.model.layers import Block, Embeddings
from src.model.tokenizer import Tokenizer


def main() -> None:
    torch.manual_seed(0)
    tok = Tokenizer()

    print("1. Tokenizer")
    for text in ["The capital of France is Paris", "Distributed systems", "unbelievably", "  2026!"]:
        ids = tok.encode(text)
        print(f"  {text!r:<34} -> {ids}")
        print(f"  {'':<34}    pieces {tok.pieces(ids)}")

    print("\n2. Shapes through embeddings and one Block (random weights)")
    cfg = GPTConfig()
    emb, block = Embeddings(cfg), Block(cfg)
    ids = torch.tensor([tok.encode("The capital of France is Paris")])
    with torch.inference_mode():
        x = emb(ids)
        a = block.attn
        q, k, v = a.c_attn(block.ln_1(x)).split(cfg.d_model, dim=-1)
        qh = a.split_heads(q)
        scores = qh @ a.split_heads(k).transpose(-2, -1)
        h1 = block.mlp.c_fc(x)
        y = block(x)
    for name, t in [("ids", ids), ("embeddings", x), ("q (one of q,k,v)", q), ("q split in heads", qh),
                    ("scores q @ k^T", scores), ("MLP inner (c_fc)", h1), ("block output", y)]:
        print(f"  {name:<18} {str(tuple(t.shape)):<18}")

    print("\n3. Parameters per piece")
    for name, module in [("embeddings wte", emb.wte), ("embeddings wpe", emb.wpe), ("attention", block.attn),
                         ("mlp", block.mlp), ("2 x LayerNorm", torch.nn.ModuleList([block.ln_1, block.ln_2])),
                         ("whole block", block)]:
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name:<16} {n:>12,}  ({n * 4 / 1e6:6.1f} MB)")


if __name__ == "__main__":
    main()
