"""
config.py - Chapter 3
Every size the model needs, in one place. The defaults are distilgpt2's.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50257   # V: number of different tokens
    n_ctx: int = 1024         # longest sequence the model has positions for
    d_model: int = 768        # d: width of every hidden state
    n_head: int = 12          # h: attention heads
    n_layer: int = 6          # L: transformer blocks (distilgpt2 has 6, gpt2 has 12)
    ln_eps: float = 1e-5      # small number added inside LayerNorm to avoid dividing by 0

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head   # 768 / 12 = 64

    @property
    def d_ff(self) -> int:
        return 4 * self.d_model              # 3072: MLP inner width
