"""
stage.py - Chapter 8
One pipeline stage: a contiguous range of GPT-2 blocks, plus the embeddings if it is the
first stage and the output layer if it is the last. Loads ONLY the weights it needs.
"""
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn

from src.model.config import GPTConfig
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_embed:
            if x.dtype != torch.int64 or x.dim() != 2:
                raise ValueError(f"first stage expects token ids [b, s] int64, got {tuple(x.shape)} {x.dtype}")
            positions = torch.arange(x.shape[1])
            x = self.wte(x) + self.wpe(positions)
        elif x.dtype != torch.float32 or x.dim() != 3 or x.shape[-1] != self.cfg.d_model:
            raise ValueError(f"stage expects hidden [b, s, {self.cfg.d_model}] float32, got {tuple(x.shape)} {x.dtype}")
        for i in range(self.first, self.last):
            x = self.blocks[str(i)](x)
        if self.has_head:
            return self.ln_f(x[:, -1:, :]) @ self.wte.weight.T      # logits of the last position only
        return x


def load_stage(first: int, last: int, has_embed: bool, has_head: bool, name: str = "distilgpt2") -> Stage:
    """Build a Stage and read just its tensors from the safetensors file."""
    stage = Stage(GPTConfig(), first, last, has_embed, has_head)
    wanted = set(stage.state_dict().keys())
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
    stage.load_state_dict(weights, strict=True)
    return stage.eval()
