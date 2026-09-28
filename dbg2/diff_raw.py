"""Diff two raw ext.reconstruct outputs (CUDA vs ROCm) and characterize the PATTERN of
wrong elements, so we can map the error back to (tile, warp, lane, bit-window).

  python diff_raw.py cuda_raw.pt rocm_raw.pt
"""
import sys
import torch


def load(p):
    d = torch.load(p, map_location="cpu")
    return d["w"].float().contiguous()


def main():
    a = load(sys.argv[1])  # reference (CUDA)
    b = load(sys.argv[2])  # ROCm under test
    assert a.shape == b.shape, (a.shape, b.shape)
    print(f"shape={tuple(a.shape)}  (a={sys.argv[1]}  b={sys.argv[2]})")

    diff = (a - b).abs()
    n_wrong = (diff > 0).sum().item()
    n = a.numel()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    print(f"cos={cos:.6f}  max_abs={diff.max().item():.6f}  mean_abs={diff.mean().item():.6f} "
          f"frac_wrong={n_wrong / n:.6f} ({n_wrong}/{n})")

    # Per-row wrong fraction (first 40 rows)
    row_frac = (diff > 0).float().mean(dim=1)
    print("\nper-row wrong fraction (row:frac):")
    for i in range(0, min(40, a.shape[0]), 4):
        print(f"  row {i:4d}: {row_frac[i].item():.4f}")

    # Per-col wrong fraction (first 40 cols)
    col_frac = (diff > 0).float().mean(dim=0)
    print("\nper-col wrong fraction (col:frac):")
    for j in range(0, min(40, a.shape[1]), 4):
        print(f"  col {j:4d}: {col_frac[j].item():.4f}")

    # Row 0, cols 0..32: the raw dequantized values side by side (pre-Hadamard)
    print("\nrow 0, cols 0..31 (a=ref  b=rocm):")
    for j in range(0, 32, 8):
        seg_a = [f"{v: .4f}" for v in a[0, j:j + 8].tolist()]
        seg_b = [f"{v: .4f}" for v in b[0, j:j + 8].tolist()]
        print(f"  c{j:2d}: a[{', '.join(seg_a)}]")
        print(f"      b[{', '.join(seg_b)}]")

    # A sample of wrong (i, j)
    idx = (diff > 0).nonzero()
    print(f"\nsample of {min(24, idx.shape[0])} wrong (i, j):  a -> b")
    for t in range(0, min(24, idx.shape[0])):
        i, j = idx[t].tolist()
        print(f"  ({i:4d},{j:4d}): {a[i, j].item(): .5f} -> {b[i, j].item(): .5f}")


if __name__ == "__main__":
    main()
