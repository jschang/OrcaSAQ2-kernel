"""#14 debug: where do MODE 3/4 partials diverge from MODE 0?"""
import json
import os
import sys

import torch
from safetensors import safe_open

from exllamav3.ext import exllamav3_ext as ext

sys.path.insert(0, os.environ.get("BENCH_BUILD_DIR", "/tmp/gemvbench"))
import gemv_bench

MODEL = os.environ.get("MODEL_PATH", "/model")
DEV = torch.device("cuda", 0)
IDX = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]

base = sys.argv[1] if len(sys.argv) > 1 else "model.language_model.layers.0.linear_attn.in_proj_qkv"


def lt(key, dtype=None):
    with safe_open(os.path.join(MODEL, IDX[key]), framework="pt", device="cpu") as f:
        t = f.get_tensor(key)
    if dtype is not None:
        t = t.to(dtype)
    return t.to(DEV).contiguous()


trellis = lt(base + ".trellis")
suh = lt(base + ".suh", torch.float16)
mul1 = base + ".mul1" in IDX and bool(lt(base + ".mul1").numel())
mcg = base + ".mcg" in IDX and bool(lt(base + ".mcg").numel())
K = trellis.shape[0] * 16
N = trellis.shape[1] * 16
tile_u16 = trellis.shape[2]
print(f"{base}: K={K} N={N} tile_u16={tile_u16} mcg={mcg} mul1={mul1}")

torch.manual_seed(0)
M = 1
x = (torch.randn(M, K, dtype=torch.float16, device=DEV) * 0.05).contiguous()
A_had = torch.empty_like(x)
ext.had_r_128(x, A_had, suh, None, 1.0)

n_chunks = N // 128
splits = max(1, (512 + n_chunks - 1) // n_chunks)
splits = min(splits, K // (16 * 16))
while (K // 16) % splits:
    splits += 1
kt_max = (K // 16) // splits
print(f"splits={splits} kt_max={kt_max}")

ref = gemv_bench.bench_partial(A_had, trellis, M, 0)
ref2 = gemv_bench.bench_partial(A_had, trellis, M, 0)
print("MODE0 vs MODE0:", (ref2 - ref).abs().max().item())

for mode in (1, 2, 3, 4):
    out = gemv_bench.bench_partial(A_had, trellis, M, mode)
    d = (out - ref).abs()
    print(f"MODE{mode}: max={d.max().item():.4g}  #nonzero={int((d > 0).sum())} / {d.numel()}")
    nz = (d > 1e-3).nonzero()
    if nz.numel():
        print("   first divergent (split, m, n):", nz[:8].tolist())
        svals = nz[:, 0].unique()
        print("   splits affected:", svals[:12].tolist(), "..." if svals.numel() > 12 else "")
        nvals = nz[:, 2]
        print("   n range:", nvals.min().item(), "-", nvals.max().item())
