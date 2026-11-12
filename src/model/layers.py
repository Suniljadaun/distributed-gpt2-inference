"""
layers.py - Chapter 3
The pieces of a GPT-2 transformer, written out by hand:
Embeddings, LayerNorm, GELU, MLP, CausalSelfAttention and Block.
Layer names (c_attn, c_proj, c_fc, ln_1, ...) match Hugging Face's, so loading weights is easy (Ch 4).
"""
import math

import torch
from torch import nn

from src.model.config import GPTConfig
from src.model.kv_cache import LayerKV


class Embeddings(nn.Module):
    """token ids [b, s] -> hidden states [b, s, d]: token meaning + position."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)   # word/token embedding table
        self.wpe = nn.Embedding(cfg.n_ctx, cfg.d_model)        # position embedding table

    def forward(self, ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        s = ids.shape[1]
        positions = torch.arange(start_pos, start_pos + s, device=ids.device)  # [s]
        return self.wte(ids) + self.wpe(positions)              # [b, s, d] + [s, d] (broadcast)


class LayerNorm(nn.Module):
    """Normalise each token's vector to mean 0, variance 1, then scale and shift."""

    def __init__(self, d: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))    # gamma, learned scale
        self.bias = nn.Parameter(torch.zeros(d))     # beta, learned shift
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)                        # [b, s, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)          # [b, s, 1]
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.weight + self.bias


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GPT-2's activation: the tanh approximation of GELU."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class MLP(nn.Module):
    """Per-token feed-forward network: d -> 4d -> GELU -> d."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.d_model, cfg.d_ff)
        self.c_proj = nn.Linear(cfg.d_ff, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(gelu(self.c_fc(x)))


class CausalSelfAttention(nn.Module):
    """Each token mixes in information from itself and earlier tokens, never later ones."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.d_head = cfg.d_head
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model)   # q, k, v in one matmul
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model)       # mix the heads back together

    def split_heads(self, t: torch.Tensor) -> torch.Tensor:
        b, s, _ = t.shape
        return t.view(b, s, self.n_head, self.d_head).transpose(1, 2)    # [b, h, s, d_h]

    def merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        b, _, s, _ = t.shape
        return t.transpose(1, 2).reshape(b, s, self.n_head * self.d_head)  # [b, s, d]

    def forward(self, x: torch.Tensor, kv: LayerKV | None = None) -> torch.Tensor:
        s = x.shape[1]
        q, k, v = self.c_attn(x).split(self.n_head * self.d_head, dim=-1)   # 3 x [b, s, d]
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)

        past = 0
        if kv is not None:                                                 # Ch 9: reuse earlier K, V
            past = kv.length
            k, v = kv.append(k, v)                                         # now [b, h, past + s, d_h]
        total = k.shape[2]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)        # [b, h, s, past + s]
        future = torch.triu(torch.ones(s, total, dtype=torch.bool, device=x.device), diagonal=past + 1)
        scores = scores.masked_fill(future, float("-inf"))                 # hide later tokens
        weights = torch.softmax(scores, dim=-1)                            # each row sums to 1

        out = weights @ v                                                  # [b, h, s, d_h]
        return self.c_proj(self.merge_heads(out))                          # [b, s, d]

    def forward_decode_batch(self, x: torch.Tensor, kvs: list[LayerKV]) -> torch.Tensor:
        """Ch 11: ONE new token for each of B different requests, x = [B, 1, d].
        The projections run once for the whole batch; attention runs per request, on its own cache."""
        q, k, v = self.c_attn(x).split(self.n_head * self.d_head, dim=-1)   # one matmul for all B
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)   # [B, h, 1, d_h]
        outs = []
        for i, kv in enumerate(kvs):
            keys, values = kv.append(k[i:i + 1], v[i:i + 1])               # this request's whole cache
            scores = (q[i:i + 1] @ keys.transpose(-2, -1)) / math.sqrt(self.d_head)   # [1, h, 1, T]
            outs.append(torch.softmax(scores, dim=-1) @ values)            # no mask: 1 new token sees all
        return self.c_proj(self.merge_heads(torch.cat(outs, dim=0)))       # [B, 1, d]


class Block(nn.Module):
    """One transformer layer: x + attn(ln_1(x)), then x + mlp(ln_2(x))."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(cfg.d_model, cfg.ln_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.d_model, cfg.ln_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, kv: LayerKV | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), kv)   # residual: add, don't replace
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_decode_batch(self, x: torch.Tensor, kvs: list[LayerKV]) -> torch.Tensor:
        """Ch 11: x = [B, 1, d], one decode token per request; kvs[i] = request i's cache for this layer."""
        x = x + self.attn.forward_decode_batch(self.ln_1(x), kvs)
        x = x + self.mlp(self.ln_2(x))        # LayerNorm and MLP work per token: batching is free
        return x
