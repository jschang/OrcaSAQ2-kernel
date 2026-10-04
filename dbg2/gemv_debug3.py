"""#14 debug 3: verify the MODE 5 staged-x writeback against the host-side x."""
import json
import os
import sys

import torch
from safetensors import safe_open

MODEL = os.environ.get("MODEL_PATH", "/model")
DEV = torch.device("cuda", 0)
IDX = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]

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

which = sys.argv[2] if len(sys.argv) > 2 else "rand"
M = 1
if which == "rand":
    torch.manual_seed(0)
    x = (torch.randn(M, K, dtype=torch.float32, device=DEV) * 0.05).to(torch.float16)
else:
    x = torch.ones(M, K, dtype=torch.float16, device=DEV)

out = gemv_bench.bench_partial(x, trellis, M, 5)  # (splits, M, N) f32; probe writes flat rows
# kernel writeback: flat[s * (M*kt_max*16) + i] for i < M*kt_max*16  (M=1)
flat = out.view(-1)
got = torch.stack([flat[s * kt_max * 16: (s + 1) * kt_max * 16] for s in range(splits)])
expected = torch.stack([x[0, s * kt_max * 16: (s + 1) * kt_max * 16].float() for s in range(splits)])
d = (got - expected).abs()
print(f"staged-x probe ({which}): max|d|={d.max().item():.4g}  #mismatch={int((d > 1e-6).sum())}")
bad = (d > 1e-6).nonzero()
if bad.numel():
    print("first mismatches (split, i):", bad[:10].tolist())
    svals = bad[:, 0].unique()
    print("splits affected:", svals[:8].tolist(), "of", splits)
    for s in svals[:3].tolist():
        ib = bad[bad[:, 0] == s][:, 1]
        print(f"  split {s}: i range {ib.min().item()}-{ib.max().item()}  count {ib.numel()}")
