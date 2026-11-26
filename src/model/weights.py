"""
weights.py - Chapter 4
Load distilgpt2's trained weights from Hugging Face's safetensors file into our GPT.
Only file reading, renaming and transposing: no Hugging Face model object is built.
"""
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from src.model.config import config_for
from src.model.gpt import GPT

TRANSPOSED = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def hf_name_to_ours(name: str) -> str | None:
    """'transformer.h.3.mlp.c_fc.weight' -> 'blocks.3.mlp.c_fc.weight'. None = skip this tensor."""
    name = name.removeprefix("transformer.")
    if name.endswith(".attn.bias"):         # HF's stored causal mask, not a weight: we build our own
        return None
    if name.startswith("h."):
        return "blocks." + name.removeprefix("h.")
    if name.startswith(("wte.", "wpe.")):
        return "emb." + name
    return name                              # ln_f.weight, ln_f.bias


def convert_state_dict(hf_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    ours = {}
    for hf_name, tensor in hf_sd.items():
        name = hf_name_to_ours(hf_name)
        if name is None:
            continue
        if name.endswith(TRANSPOSED):
            tensor = tensor.T               # HF Conv1D [in, out] -> nn.Linear [out, in]
        ours[name] = tensor.contiguous()
    return ours


def load_pretrained(name: str = "distilgpt2") -> GPT:
    """Build our GPT and fill it with the trained weights. Returns the model in eval mode."""
    path = hf_hub_download(name, "model.safetensors")     # downloads once, then uses the cache
    model = GPT(config_for(name))
    model.load_state_dict(convert_state_dict(load_file(path)), strict=True)
    return model.eval()
