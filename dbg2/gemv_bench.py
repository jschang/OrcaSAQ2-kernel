"""#14: GEMV partial kernel cost breakdown -- MODE variants.

For each of the four representative decode shapes:
  - time MODE 0..4 of the bench variant kernel (kernel-only, torch.profiler)
  - cross-check MODE 3/4 partials against MODE 0 (max abs diff, fp32 partials)
  - effective packed-read TB/s per MODE

Modes: 0=production, 1=no-dequant, 2=no-FMA, 3=x-in-smem, 4=x-in-smem+FMA2.
Run: python3 /work/dbg2/gemv_bench.py [M]
"""
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
Ms = [int(sys.argv[1])] if len(sys.argv) > 1 else [1, 16]

MODE_NAMES = {0: "prod   ", 1: "no-dq  ", 2: "no-fma ", 3: "x-smem ", 4: "xsmem+f2"}


def lt(key, dtype=None):
    with safe_open(os.path.join(MODEL, IDX[key]), framework="pt", device="cpu") as f:
        t = f.get_tensor(key)
    if dtype is not None:
        t = t.to(dtype)
    return t.to(DEV).contiguous()


def profile_one(fn, iters=100):
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    us = 0.0
    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.self_device_time_total > 0:
            assert "variant" in evt.key, f"unexpected kernel {evt.key}"
            us = evt.self_device_time_total / iters
    return us


SHAPES = {
    "qkv (5120x10240)": "model.language_model.layers.0.linear_attn.in_proj_qkv",
    "gate (5120x17408)": "model.language_model.layers.0.mlp.gate_proj",
    "down (17408x5120)": "model.language_model.layers.0.mlp.down_proj",
    "q    (5120x12288)": "model.language_model.layers.3.self_attn.q_proj",
}

for name, base in SHAPES.items():
    trellis = lt(base + ".trellis")
    suh = lt(base + ".suh", torch.float16)
    svh = lt(base + ".svh", torch.float16)
    mul1 = base + ".mul1" in IDX and bool(lt(base + ".mul1").numel())
    mcg = base + ".mcg" in IDX and bool(lt(base + ".mcg").numel())
    assert mul1 and not mcg, f"{name}: bench needs mul1-only"

    K = trellis.shape[0] * 16
    N = trellis.shape[1] * 16
    packed_bytes = trellis.numel() * 2

    for M in Ms:
        x = (torch.randn(M, K, dtype=torch.float16, device=DEV) * 0.05).contiguous()
        A_had = torch.empty_like(x)
        ext.had_r_128(x, A_had, suh, None, 1.0)

        ref = gemv_bench.bench_partial(A_had, trellis, M, 0)
        times = {}
        for mode in (0, 1, 2, 3, 4):
            times[mode] = profile_one(lambda m=mode: gemv_bench.bench_partial(A_had, trellis, M, m))

        d3 = (gemv_bench.bench_partial(A_had, trellis, M, 3) - ref).abs().max().item()
        d4 = (gemv_bench.bench_partial(A_had, trellis, M, 4) - ref).abs().max().item()

        print(f"\n== {name}  M={M}  packed={packed_bytes/1e6:.1f} MB ==")
        for mode in (0, 1, 2, 3, 4):
            t = times[mode]
            bw = packed_bytes / (t * 1e-6) / 1e12
            print(f"   {MODE_NAMES[mode]}  {t:8.1f} us   {bw:5.2f} TB/s")
        print(f"   diffs vs MODE0: mode3 max|d|={d3:.2e}  mode4 max|d|={d4:.2e}  (|C_partial| scale ~1e1)")
        del trellis, suh, svh, x, A_had, ref
        torch.cuda.empty_cache()

print("\ndone")
