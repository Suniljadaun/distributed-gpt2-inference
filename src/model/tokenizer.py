"""
tokenizer.py - Chapter 3
Text <-> token ids. A thin wrapper around GPT-2's byte-level BPE tokenizer from Hugging Face,
so the rest of our code depends on two methods only, not on the transformers library.
"""
from transformers import GPT2TokenizerFast


class Tokenizer:
    def __init__(self, name: str = "distilgpt2") -> None:
        self._tok = GPT2TokenizerFast.from_pretrained(name)
        self.eos_id: int = self._tok.eos_token_id   # 50256, "<|endoftext|>"
        self.vocab_size: int = len(self._tok)       # 50257

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    def pieces(self, ids: list[int]) -> list[str]:
        """Each id as its own text piece, to see where the tokenizer cut the text."""
        return [self._tok.decode([i]) for i in ids]
