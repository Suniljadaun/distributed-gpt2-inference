"""
test_ch03.py - Chapter 3
Each piece of our transformer, checked against Hugging Face's distilgpt2 with the same weights.
Run:  python -m pytest tests/test_ch03.py -v
"""
import pytest
import torch
from transformers import GPT2Model

from src.model.config import GPTConfig
from src.model.layers import MLP, Block, CausalSelfAttention, Embeddings, LayerNorm, gelu
from src.model.tokenizer import Tokenizer

PROMPT = "The capital of France is Paris"
ATOL = 1e-4   # floats will differ in the last digits (Ch 2.2.8); this is "the same"


@pytest.fixture(scope="module")
def hf():
    """Load Hugging Face's distilgpt2 once for the whole file."""
    return GPT2Model.from_pretrained("distilgpt2").eval()


@pytest.fixture(scope="module")
def ids() -> torch.Tensor:
    return torch.tensor([Tokenizer().encode(PROMPT)])


def copy_linear(ours: torch.nn.Linear, theirs) -> None:
    """HF's Conv1D stores weight as [in, out]; nn.Linear wants [out, in]. Transpose."""
    ours.weight.copy_(theirs.weight.T)
    ours.bias.copy_(theirs.bias)


def copy_ln(ours: LayerNorm, theirs) -> None:
    ours.weight.copy_(theirs.weight)
    ours.bias.copy_(theirs.bias)


@torch.inference_mode()
def test_tokenizer_round_trip() -> None:
    tok = Tokenizer()
    ids = tok.encode(PROMPT)
    assert ids == [464, 3139, 286, 4881, 318, 6342]
    assert tok.decode(ids) == PROMPT
    assert tok.vocab_size == 50257 and tok.eos_id == 50256


@torch.inference_mode()
def test_embeddings(hf, ids) -> None:
    emb = Embeddings(GPTConfig())
    emb.wte.weight.copy_(hf.wte.weight)
    emb.wpe.weight.copy_(hf.wpe.weight)
    expected = hf(ids, output_hidden_states=True).hidden_states[0]   # input to block 0
    assert torch.allclose(emb(ids), expected, atol=ATOL)


@torch.inference_mode()
def test_layernorm_matches_torch() -> None:
    x = torch.randn(2, 5, 768) * 3 + 7
    ours = LayerNorm(768, 1e-5)
    ref = torch.nn.LayerNorm(768, eps=1e-5)
    y = ours(x)
    assert torch.allclose(y, ref(x), atol=ATOL)
    assert torch.allclose(y.mean(-1), torch.zeros(2, 5), atol=1e-4)   # mean 0 per token


@torch.inference_mode()
def test_gelu_matches_torch() -> None:
    x = torch.linspace(-6, 6, 1001)
    assert torch.allclose(gelu(x), torch.nn.functional.gelu(x, approximate="tanh"), atol=1e-6)


@torch.inference_mode()
def test_mlp(hf) -> None:
    mlp = MLP(GPTConfig())
    copy_linear(mlp.c_fc, hf.h[0].mlp.c_fc)
    copy_linear(mlp.c_proj, hf.h[0].mlp.c_proj)
    x = torch.randn(1, 6, 768)
    assert torch.allclose(mlp(x), hf.h[0].mlp(x), atol=ATOL)


@torch.inference_mode()
def test_attention(hf, ids) -> None:
    attn = CausalSelfAttention(GPTConfig())
    copy_linear(attn.c_attn, hf.h[0].attn.c_attn)
    copy_linear(attn.c_proj, hf.h[0].attn.c_proj)

    seen = {}   # capture what HF's attention received and returned during a real forward pass

    def hook(module, args, kwargs, output):
        seen["x"] = args[0] if args else kwargs["hidden_states"]
        seen["y"] = output[0] if isinstance(output, tuple) else output

    handle = hf.h[0].attn.register_forward_hook(hook, with_kwargs=True)
    hf(ids)
    handle.remove()
    assert torch.allclose(attn(seen["x"]), seen["y"], atol=ATOL)


@torch.inference_mode()
def test_attention_is_causal() -> None:
    """Changing a LATER token must not change the output at EARLIER positions."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(GPTConfig())
    x = torch.randn(1, 6, 768)
    x2 = x.clone()
    x2[:, 4:] = torch.randn(1, 2, 768)        # change tokens 4 and 5 only
    y, y2 = attn(x), attn(x2)
    assert torch.allclose(y[:, :4], y2[:, :4], atol=1e-6)   # positions 0-3 unaffected
    assert not torch.allclose(y[:, 4:], y2[:, 4:])          # 4-5 did change


@torch.inference_mode()
@pytest.mark.parametrize("layer", [0, 5])
def test_block(hf, ids, layer) -> None:
    cfg = GPTConfig()
    block = Block(cfg)
    theirs = hf.h[layer]
    copy_ln(block.ln_1, theirs.ln_1)
    copy_ln(block.ln_2, theirs.ln_2)
    copy_linear(block.attn.c_attn, theirs.attn.c_attn)
    copy_linear(block.attn.c_proj, theirs.attn.c_proj)
    copy_linear(block.mlp.c_fc, theirs.mlp.c_fc)
    copy_linear(block.mlp.c_proj, theirs.mlp.c_proj)

    hidden = hf(ids, output_hidden_states=True).hidden_states
    ours = block(hidden[layer])
    if layer == cfg.n_layer - 1:
        ours = hf.ln_f(ours)   # HF's LAST hidden state already has the final LayerNorm applied
    assert torch.allclose(ours, hidden[layer + 1], atol=ATOL)
