"""
kv_cache.py - Chapter 9
The keys and values of tokens already processed, kept between decode steps.
LayerKV: one layer's K and V. KVCache: all the layers one process holds, for ONE request.
"""
import torch


class LayerKV:
    """K and V of one attention layer, shape [b, h, T, d_head], growing by one step at a time."""

    def __init__(self) -> None:
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add this step's K, V (new positions only) and return K, V for ALL positions so far."""
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    @property
    def nbytes(self) -> int:
        if self.k is None:
            return 0
        return (self.k.nelement() + self.v.nelement()) * self.k.element_size()


class KVCache:
    """Every layer's LayerKV for one request, keyed by the GLOBAL layer index."""

    def __init__(self) -> None:
        self.layers: dict[int, LayerKV] = {}

    def layer(self, i: int) -> LayerKV:
        if i not in self.layers:
            self.layers[i] = LayerKV()
        return self.layers[i]

    @property
    def length(self) -> int:
        """Number of positions cached (the same in every layer)."""
        lengths = {kv.length for kv in self.layers.values()}
        if len(lengths) > 1:
            raise RuntimeError(f"layers disagree about cache length: {sorted(lengths)}")
        return lengths.pop() if lengths else 0

    @property
    def nbytes(self) -> int:
        return sum(kv.nbytes for kv in self.layers.values())
