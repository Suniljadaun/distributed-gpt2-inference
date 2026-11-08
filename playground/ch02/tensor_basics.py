"""
tensor_basics.py - Chapter 2, demo 1
Tensors: shape, dtype, memory size, views, transpose and indexing.
Run:  python playground/ch02/tensor_basics.py
"""
import torch


def show(name: str, t: torch.Tensor) -> None:
    """Print a tensor's shape, dtype and how many bytes its data takes."""
    nbytes = t.nelement() * t.element_size()
    print(f"  {name:<12} shape={str(tuple(t.shape)):<16} dtype={str(t.dtype):<15} bytes={nbytes:,}")


def main() -> None:
    torch.manual_seed(0)  # same "random" numbers every run

    print("1. Creating tensors")
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    show("a", a)
    show("zeros", torch.zeros(2, 3))
    show("randn", torch.randn(2, 3))
    show("arange", torch.arange(6))
    show("token ids", torch.tensor([[464, 3139, 318]]))

    print("\n2. The hidden state that travels between our workers")
    h_token = torch.randn(1, 1, 768)    # [batch, seq, d_model]: one new token
    h_prompt = torch.randn(1, 20, 768)  # a 20-token prompt
    show("1 token", h_token)
    show("20 tokens", h_prompt)
    show("1 token fp16", h_token.half())

    print("\n3. Views, reshape and transpose")
    x = torch.arange(12)
    m = x.view(3, 4)
    print("  x          ", x.tolist())
    print("  x.view(3,4)", m.tolist())
    print("  m.T        ", m.T.tolist())
    print("  contiguous?  m:", m.is_contiguous(), "  m.T:", m.T.is_contiguous())
    m[0, 0] = 100
    print("  after m[0,0] = 100, x[0] =", x[0].item(), "(a view shares memory)")
    try:
        m.T.view(12)
    except RuntimeError as e:
        print("  m.T.view(12) fails:", str(e)[:60], "...")
    print("  m.T.reshape(12)", m.T.reshape(12).tolist(), "(reshape copies when it must)")

    print("\n4. Splitting 768 into 12 heads of 64 (preview of Chapter 3)")
    heads = h_prompt.view(1, 20, 12, 64).transpose(1, 2)
    show("heads", heads)

    print("\n5. Indexing and slicing")
    t = torch.arange(24).view(2, 3, 4)
    print("  t[0]       ", tuple(t[0].shape))
    print("  t[:, -1]   ", tuple(t[:, -1].shape), " <- last position of each sequence")
    print("  t[..., :2] ", tuple(t[..., :2].shape))
    print("  t[:, -1:]  ", tuple(t[:, -1:].shape), " <- same, but keeps the seq dimension")


if __name__ == "__main__":
    main()
