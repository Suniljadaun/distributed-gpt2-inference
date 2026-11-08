"""
matmul_shapes.py - Chapter 2, demo 2
Matrix multiply, broadcasting, a linear layer by hand, softmax and floating-point surprises.
Run:  python playground/ch02/matmul_shapes.py
"""
import torch


def main() -> None:
    torch.manual_seed(0)

    print("1. Matrix multiply: [n, k] @ [k, m] -> [n, m]")
    A = torch.randn(2, 3)
    B = torch.randn(3, 4)
    C = A @ B
    print("  A @ B shape:", tuple(C.shape))
    by_hand = sum(A[0, k] * B[k, 1] for k in range(3))
    print(f"  C[0,1] = {C[0, 1].item():.4f}   by hand = {by_hand.item():.4f}")
    try:
        A @ A
    except RuntimeError as e:
        print("  A @ A fails:", e)

    print("\n2. Batched matmul: leading dimensions ride along")
    q = torch.randn(1, 12, 20, 64)  # [batch, heads, seq, head_dim]
    k = torch.randn(1, 12, 20, 64)
    scores = q @ k.transpose(-2, -1)
    print("  q @ k^T shape:", tuple(scores.shape), " <- a 20x20 score table per head")

    print("\n3. Broadcasting")
    x = torch.randn(1, 20, 768)
    bias = torch.randn(768)
    print("  (1,20,768) + (768,) ->", tuple((x + bias).shape))
    col = torch.tensor([[1.0], [2.0], [3.0]])
    row = torch.tensor([10.0, 20.0])
    print("  (3,1) + (2,) ->", (col + row).tolist())
    try:
        torch.randn(3, 4) + torch.randn(3)
    except RuntimeError as e:
        print("  (3,4) + (3,) fails:", e)

    print("\n4. A linear layer is matmul + bias")
    lin = torch.nn.Linear(768, 3072)
    with torch.no_grad():
        mine = x @ lin.weight.T + lin.bias
        theirs = lin(x)
    n_params = sum(p.numel() for p in lin.parameters())
    print("  weight shape:", tuple(lin.weight.shape), "(out, in)")
    print("  output shape:", tuple(mine.shape))
    print(f"  parameters  : {n_params:,}  ({n_params * 4 / 1e6:.1f} MB in fp32)")
    print("  same as nn.Linear?  allclose:", torch.allclose(mine, theirs, atol=1e-5),
          "  exactly equal:", torch.equal(mine, theirs))

    print("\n5. Softmax turns scores into probabilities")
    logits = torch.tensor([2.0, 1.0, 0.1])
    probs = torch.softmax(logits, dim=-1)
    print("  probs:", [round(p, 3) for p in probs.tolist()], " sum:", round(probs.sum().item(), 3))
    print("  greedy pick (argmax):", probs.argmax().item())
    big = torch.tensor([1000.0, 999.0])
    naive = big.exp() / big.exp().sum()
    print("  naive softmax of [1000, 999]:", naive.tolist())
    print("  torch.softmax of [1000, 999]:", [round(p, 3) for p in torch.softmax(big, dim=-1).tolist()])

    print("\n6. Floating point: order of addition matters")
    one = torch.tensor(1.0)
    big1 = torch.tensor(1e8)
    print("  (1e8 + 1) - 1e8 =", ((big1 + one) - big1).item(), "in float32")
    v = torch.randn(1_000_000)
    s1 = v.sum()
    s2 = v[torch.randperm(v.numel())].sum()
    print(f"  sum in order   : {s1.item():.6f}")
    print(f"  sum shuffled   : {s2.item():.6f}")
    print("  torch.equal:", torch.equal(s1, s2), "  allclose:", torch.allclose(s1, s2, atol=1e-3))


if __name__ == "__main__":
    main()
