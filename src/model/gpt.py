"""
gpt.py - Chapter 4
The whole GPT-2: embeddings -> n_layer Blocks -> final LayerNorm -> output layer (tied to wte).
Split into three stages (embed / run_blocks / head) because Chapter 8 runs them in different processes.
"""
import torch
from torch import nn

from src.model.config import GPTConfig
from src.model.layers import Block, Embeddings, LayerNorm
from src.model.kv_cache import KVCache 

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.d_model, cfg.ln_eps)

    def embed(self, ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Stage 1: token ids [b, s] -> hidden states [b, s, d]."""
        return self.emb(ids, start_pos)

    def run_blocks(self, x: torch.Tensor, first: int, last: int, cache: KVCache | None = None) -> torch.Tensor:
        """Stage 2: run Blocks first..last-1 (Python-range style). [b, s, d] -> [b, s, d]."""
        for i in range(first, last):
            x = self.blocks[i](x, cache.layer(i) if cache is not None else None)
        return x


    def head(self, x: torch.Tensor) -> torch.Tensor:
        """Stage 3: hidden states [b, s, d] -> logits [b, s, V], one score per vocabulary token."""
        return self.ln_f(x) @ self.emb.wte.weight.T     # weight tying: reuse the embedding table

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        x = self.run_blocks(x, 0, self.cfg.n_layer)
        return self.head(x)
