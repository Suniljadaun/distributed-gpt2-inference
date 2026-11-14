"""
stage.py - Chapter 8 (Ch 9: start_pos + KV-cache; Ch 11: batched decode step; Ch 15: blocks can move)
One pipeline stage: a contiguous range of GPT-2 blocks, plus the embeddings if it is the
first stage and the output layer if it is the last. Loads ONLY the weights it needs.
Ch 15: the range can change while running (add_block / drop_block), and a slowdown can be injected.
"""
import threading
import time

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn

from src.model.config import GPTConfig
from src.model.kv_cache import KVCache
from src.model.layers import Block, LayerNorm
from src.model.weights import TRANSPOSED, hf_name_to_ours


class Stage(nn.Module):
    def __init__(self, cfg: GPTConfig, first: int, last: int, has_embed: bool, has_head: bool) -> None:
        super().__init__()
        if not 0 <= first <= last <= cfg.n_layer:
            raise ValueError(f"bad layer range {first}..{last}")
        self.cfg, self.first, self.last = cfg, first, last
        self.has_embed, self.has_head = has_embed, has_head
        if has_embed or has_head:
            self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)   # input lookup AND tied output layer
        if has_embed:
            self.wpe = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.blocks = nn.ModuleDict({str(i): Block(cfg) for i in range(first, last)})
        if has_head:
            self.ln_f = LayerNorm(cfg.d_model, cfg.ln_eps)
        self.block_delay_ms = 0.0       # Ch 15 fault: every block takes this much longer ...
        self.slow_from = 0.0            # ... from this time.monotonic() on (a slower machine)
        self._timing = threading.local()   # Ch 15: per-thread time spent in blocks during the last forward

    @property
    def last_blocks_ms(self) -> float:
        """Time the CALLING thread's last forward spent in blocks (what moves when a layer moves)."""
        return getattr(self._timing, "blocks_ms", 0.0)

    def _run_block(self, i: int, fn) -> torch.Tensor:
        """Run block i via fn(); with the injected fault, then wait a fixed extra delay."""
        start = time.perf_counter()
        out = fn(self.blocks[str(i)])
        if self.block_delay_ms and time.monotonic() >= self.slow_from:
            time.sleep(self.block_delay_ms / 1000)
        self._timing.blocks_ms += (time.perf_counter() - start) * 1000
        return out

    # ---------------- Ch 15: the range can change ----------------

    def add_block(self, i: int, block: Block) -> None:
        if i == self.last and not self.has_head:
            self.last += 1                                  # grow at the end (a first stage)
        elif i == self.first - 1 and not self.has_embed:
            self.first -= 1                                 # grow at the front (a last stage)
        else:
            raise ValueError(f"layer {i} is not adjacent to {self.first}..{self.last - 1} on this side")
        self.blocks[str(i)] = block

    def drop_block(self, i: int) -> None:
        if i == self.last - 1 and not self.has_head:
            self.last -= 1
        elif i == self.first and not self.has_embed:
            self.first += 1
        else:
            raise ValueError(f"layer {i} is not the movable boundary of {self.first}..{self.last - 1}")
        del self.blocks[str(i)]

    # ---------------- forward passes ----------------

    def forward(self, x: torch.Tensor, start_pos: int = 0, cache: KVCache | None = None) -> torch.Tensor:
        if self.has_embed:
            if x.dtype != torch.int64 or x.dim() != 2:
                raise ValueError(f"first stage expects token ids [b, s] int64, got {tuple(x.shape)} {x.dtype}")
            if start_pos + x.shape[1] > self.cfg.n_ctx:
                raise ValueError(f"positions {start_pos}..{start_pos + x.shape[1] - 1} exceed n_ctx {self.cfg.n_ctx}")
            positions = torch.arange(start_pos, start_pos + x.shape[1])
            x = self.wte(x) + self.wpe(positions)
        elif x.dtype != torch.float32 or x.dim() != 3 or x.shape[-1] != self.cfg.d_model:
            raise ValueError(f"stage expects hidden [b, s, {self.cfg.d_model}] float32, got {tuple(x.shape)} {x.dtype}")
        self._timing.blocks_ms = 0.0
        for i in range(self.first, self.last):
            x = self._run_block(i, lambda blk: blk(x, cache.layer(i) if cache is not None else None))
        if self.has_head:
            return self.ln_f(x[:, -1:, :]) @ self.wte.weight.T      # logits of the last position only
        return x

    def forward_decode_batch(self, x: torch.Tensor, start_positions: list[int], caches: list[KVCache]) -> torch.Tensor:
        """Ch 11: one decode step for B requests at once. x = ids [B, 1] (first stage) or hidden [B, 1, d]."""
        if self.has_embed:
            if x.dtype != torch.int64 or x.shape != (len(caches), 1):
                raise ValueError(f"first stage expects ids [B, 1] int64, got {tuple(x.shape)} {x.dtype}")
            if max(start_positions) >= self.cfg.n_ctx:
                raise ValueError(f"position {max(start_positions)} exceeds n_ctx {self.cfg.n_ctx}")
            x = self.wte(x) + self.wpe(torch.tensor(start_positions).unsqueeze(1))   # [B,1,d] + [B,1,d]
        elif x.dtype != torch.float32 or x.shape != (len(caches), 1, self.cfg.d_model):
            raise ValueError(f"stage expects hidden [B, 1, {self.cfg.d_model}] float32, got {tuple(x.shape)} {x.dtype}")
        self._timing.blocks_ms = 0.0
        for i in range(self.first, self.last):
            x = self._run_block(i, lambda blk: blk.forward_decode_batch(x, [c.layer(i) for c in caches]))
        if self.has_head:
            return self.ln_f(x) @ self.wte.weight.T                 # [B, 1, V]
        return x


def _read_weights(wanted: set[str], name: str) -> dict[str, torch.Tensor]:
    """Read only the tensors whose (Stage-style) names are in `wanted` from the safetensors file."""
    path = hf_hub_download(name, "model.safetensors")
    weights = {}
    with safe_open(path, framework="pt") as f:          # reads the header; tensors are loaded on request
        for hf_name in f.keys():
            ours = hf_name_to_ours(hf_name)              # e.g. "emb.wte.weight", "blocks.3.mlp.c_fc.weight"
            if ours is None:
                continue
            ours = ours.removeprefix("emb.")             # a Stage keeps wte/wpe at the top level
            if ours not in wanted:
                continue                                 # another worker's layer: never read it
            t = f.get_tensor(hf_name)
            weights[ours] = (t.T if ours.endswith(TRANSPOSED) else t).contiguous()
    return weights


def load_stage(first: int, last: int, has_embed: bool, has_head: bool, name: str = "distilgpt2") -> Stage:
    """Build a Stage and read just its tensors from the safetensors file."""
    stage = Stage(GPTConfig(), first, last, has_embed, has_head)
    stage.load_state_dict(_read_weights(set(stage.state_dict().keys()), name), strict=True)
    return stage.eval()


def load_block(i: int, name: str = "distilgpt2") -> Block:
    """Ch 15: one Block with layer i's trained weights, for a worker that is about to receive layer i."""
    block = Block(GPTConfig())
    prefix = f"blocks.{i}."
    weights = _read_weights({prefix + k for k in block.state_dict()}, name)
    block.load_state_dict({k.removeprefix(prefix): t for k, t in weights.items()}, strict=True)
    return block.eval()
