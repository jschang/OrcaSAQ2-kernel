"""#14 debug 2: isolate the MODE 3 x-staging with structured x patterns."""
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
K = trellis.shape[0] * 16
N = trellis.shape[1] * 16

# NOTE: skip the input Hadamard -- feed A_had directly as a raw pattern
# (bench_partial takes pre-Hadamard x; the pattern itself is the test).
M = 1

def run(pattern_name):
    if pattern_name == "ones":
        x = torch.ones(M, K, dtype=torch.float16, device=DEV)
    elif pattern_name == "ramp":
        x = torch.arange(K, dtype=torch.float32, device=DEV).unsqueeze(0).to(torch.float16)
    elif pattern_name == "rand":
        torch.manual_seed(0)
        x = (torch.randn(M, K, dtype=torch.float32, device=DEV) * 0.05).to(torch.float16)
    ref = gemv_bench.bench_partial(x, trellis, M, 0)
    m3 = gemv_bench.bench_partial(x, trellis, M, 3)
    d = (m3 - ref).abs()
    print(f"{pattern_name:>6} mode3: max|d|={d.max().item():.4g}  rel={d.max().item() / max(ref.abs().max().item(), 1e-9):.2e}  |ref|max={ref.abs().max().item():.3g}")
    m4 = gemv_bench.bench_partial(x, trellis, M, 4)
    d = (m4 - ref).abs()
    print(f"{pattern_name:>6} mode4: max|d|={d.max().item():.4g}  rel={d.max().item() / max(ref.abs().max().item(), 1e-9):.2e}  |ref|max={ref.abs().max().item():.3g}")

for p in ("ones", "ramp", "rand"):
    run(p)
