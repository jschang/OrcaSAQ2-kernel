"""#14 debug 6: which x value does the smem-x FMA load actually use per tile?

MODE 9 per-tile deltas for CTA (0,0) warp 0:
  flat[(kt*32+t)*8 + 0] = d0 (HBM x)  [+2] = d3 (smem x)
Candidates for d3's x source: tile 0, tile kt-1, tile kt (expected), tile kt+1,
and a linear-drift offset (address increment wrong).
"""
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

w_raw = torch.empty((K, N), dtype=torch.float16, device=DEV)
ext.reconstruct(w_raw, trellis, 3, False, True)
W = w_raw.float()  # (K, N)
torch.manual_seed(0)
M = 1
x = (torch.randn(M, K, dtype=torch.float32, device=DEV) * 0.05).to(torch.float16)

out = gemv_bench.bench_partial(x, trellis, M, 9)
flat = out.view(-1)
d3t = torch.stack([flat[(kt * 32 + t) * 8 + 2] for kt in range(kt_max) for t in range(32)]).view(kt_max, 32)

xf = x[0].float()
# weights for column t (t < 32), rows kt*16 + r (r in 0..7)
rows_t = torch.arange(kt_max, device=DEV).view(-1, 1) * 16 + torch.arange(8, device=DEV).view(1, -1)
wk = W[rows_t, 0:32].float()  # (kt, 8, 32)


def with_x(xsrc):  # xsrc: (kt, 8)
    return (xsrc.unsqueeze(-1) * wk).sum(dim=1)  # (kt, 32)


def show(name, cand):
    d = (cand - d3t).abs().max().item()
    print(f"  {name}: max|d|={d:.4g}")
    return d


r = rows_t  # (kt, 8)
x0 = xf[r[0]]            # tile 0 rows for all kt
xprev = xf[torch.clamp(r - 16, min=0)]
xnext = xf[torch.clamp(r + 16, max=K - 1)]
xcur = xf[r]
show("x of tile 0 (stale at kt=0)", with_x(x0))
show("x of tile kt-1 (prev)", with_x(xprev))
show("x of tile kt (expected)", with_x(xcur))
show("x of tile kt+1 (next)", with_x(xnext))

# drift from origin: addr(kt, rr) = kt*inc + rr (rr in 0..7 for khalf=0)
rr = r % 16  # (kt, 8)
for inc in (1, 2, 4, 8, 32):
    xr = xf[torch.clamp(torch.arange(kt_max, device=DEV).view(-1, 1) * inc + rr, max=K - 1)]
    show(f"drift from origin inc={inc}", with_x(xr))

# maybe x is fine but scaled per tile: fit scale factor per tile
ref = with_x(xcur)
ratio = (d3t / ref.clamp_min(1e-6))
print("per-tile |d3/ref| median (first 10):", [f"{ratio[kt].median().item():.3g}" for kt in range(10)])
