"""#14 debug 4: what x values does MODE 3's FMA actually read? (MODE 8 dump, CTA (0,0), kt=0)"""
import json
import os
import sys

import torch
from safetensors import safe_open

MODEL = os.environ.get("MODEL_PATH", "/model")
DEV = torch.device("cuda", 0)
IDX = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))[
    "weight_map"]

base = sys.argv[1] if len(sys.argv) > 1 else "model.language_model.layers.0.linear_attn.in_proj_qkv"
which = sys.argv[2] if len(sys.argv) > 2 else "rand"
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
M = 1
if which == "rand":
    torch.manual_seed(0)
    x = (torch.randn(M, K, dtype=torch.float32, device=DEV) * 0.05).to(torch.float16)
else:
    x = torch.ones(M, K, dtype=torch.float16, device=DEV)

out = gemv_bench.bench_partial(x, trellis, M, 8)  # (splits, M, N); dump at flat[0:4096]
dump = out.view(-1)[0: 256 * 16].view(256, 16)
# thread t reads rows khalf*8..khalf*8+8, stored at dump[t][rr - khalf*8]
want = x[0, 0:16].float()
bad = 0
first = None
for t in range(256):
    khalf = t // 128
    for r in range(8):
        got = dump[t, r].item()
        exp = want[khalf * 8 + r].item()
        if abs(got - exp) > 1e-6:
            bad += 1
            if first is None:
                first = (t, khalf, r, got, exp)
print(f"MODE8 FMA x-read dump ({which}): #bad={bad}/2048  first={first}")
if bad:
    # summarize by row
    byrow = {}
    for t in range(256):
        khalf = t // 128
        for r in range(8):
            got = dump[t, r].item()
            rr = khalf * 8 + r
            if abs(got - want[rr].item()) > 1e-6:
                byrow.setdefault(rr, 0)
                byrow[rr] += 1
    print("rows with bad reads:", {k: v for k, v in sorted(byrow.items())})
