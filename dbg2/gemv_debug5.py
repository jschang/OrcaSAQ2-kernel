"""#14 debug 5: MODE 9 dual FMA -- acc0 (HBM x) vs acc3 (smem x) vs python reference, per thread."""
import json
import os
import sys

import torch
from safetensors import safe_open

from exllamav3.ext import exllamav3_ext as ext

MODEL = os.environ.get("MODEL_PATH", "/model")
DEV = torch.device("cuda", 0)
IDX = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))[
    "weight_map"]
base = sys.argv[1] if len(sys.argv) > 1 else "model.language_model.layers.0.linear_attn.in_proj_qkv"
sys.path.insert(0, os.environ.get("BENCH_BUILD_DIR", "/tmp/gemvbench"))
import gemv_bench


def lt(key, dtype=None):
    with safe_open(os.path.join(MODEL, IDX[key]), framework="pt", device="cpu") as f:
        t = f.get_tensor(key)
    if dtype is not None:
        t = t.to(dtype)
    return t.to(DEV).contiguous()


trellis = lt(base + ".trellis")
K = trellis.shape[0] * 16
N = trellis.shape[1] * 16
n_chunks = N // 128
splits = max(1, (512 + n_chunks - 1) // n_chunks)
splits = min(splits, K // (16 * 16))
while (K // 16) % splits:
    splits += 1
kt_max = (K // 16) // splits
print(f"K={K} N={N} splits={splits} kt_max={kt_max}")

# RAW dequantized weight (in, out) -- no Hadamards/scales; the partial is pre-Hadamard.
w_raw = torch.empty((K, N), dtype=torch.float16, device=DEV)
ext.reconstruct(w_raw, trellis, 3, False, True)
W = w_raw.float()  # (K, N): W[k, n]
torch.manual_seed(0)
M = 1
x = (torch.randn(M, K, dtype=torch.float32, device=DEV) * 0.05).to(torch.float16)

out = gemv_bench.bench_partial(x, trellis, M, 9)
flat = out.view(-1)

# Per-tile reference: thread t (t < 32) owns column t, khalf=0 -> rows kt*16 + r, r in 0..7
rows_t = (torch.arange(kt_max, device=DEV).view(-1, 1) * 16 + torch.arange(8, device=DEV).view(1, -1))  # (kt, 8)
xk = x[0].float()[rows_t]                        # (kt, 8)
wk = W[rows_t, 0:32].float()                     # (kt, 8, 32)
exp_tile = (xk.unsqueeze(-1) * wk).sum(dim=1)    # (kt, 32)

d0t = torch.stack([flat[(kt * 32 + t) * 8] for kt in range(kt_max) for t in range(32)]).view(kt_max, 32)
d3t = torch.stack([flat[(kt * 32 + t) * 8 + 2] for kt in range(kt_max) for t in range(32)]).view(kt_max, 32)
e0 = (d0t - exp_tile).abs()
e3 = (d3t - exp_tile).abs()
print(f"per-tile d0 (HBM) vs pyref: max|d|={e0.max().item():.4g}  (|exp_tile|max={exp_tile.abs().max().item():.3g})")
print(f"per-tile d3 (smem) vs pyref: max|d|={e3.max().item():.4g}")
badkt = (e3.max(dim=1).values > 1e-6).nonzero().view(-1)
print("bad tiles (d3):", badkt.tolist() if badkt.numel() else "none")
if badkt.numel():
    for kt in badkt.tolist():
        badt = (e3[kt] > 1e-6).nonzero().view(-1)
        print(f"  kt={kt}: bad cols {badt.tolist()[:12]} (of 32)")

# Full-loop accumulators
exp_full = exp_tile.sum(dim=0)  # (32,)
acc0 = torch.stack([flat[20480 + t * 32] for t in range(32)])
acc3 = torch.stack([flat[20480 + t * 32 + 8] for t in range(32)])
e0f = (acc0 - exp_full).abs()
e3f = (acc3 - exp_full).abs()
print(f"full-loop acc0 vs pyref: max|d|={e0f.max().item():.4g}")
print(f"full-loop acc3 vs pyref: max|d|={e3f.max().item():.4g}  (|exp_full|max={exp_full.abs().max().item():.3g})")
