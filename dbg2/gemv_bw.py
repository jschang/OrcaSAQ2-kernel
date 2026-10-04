"""#14: per-shape GEMV partial-kernel bandwidth table + reference read BW.

For the four representative decode shapes:
  - exact packed trellis bytes (trellis.numel() * 2)
  - host-side split/CTA computation (mirrors exl3_gemv)
  - kernel-only time (torch.profiler) split across had_in / gemv_partial / gemv_finalize
  - effective packed-read TB/s for the partial kernel
  - reference streaming read BW of the same bytes (torch copy into a preallocated buffer)

Run inside the dev container:  python3 /work/dbg2/gemv_bw.py
"""
import json
import os

import torch
from safetensors import safe_open

from exllamav3.ext import exllamav3_ext as ext

MODEL = os.environ.get("MODEL_PATH", "/model")
DEV = torch.device("cuda", 0)
IDX = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]

GEMV_N_TILE = 128
GEMV_K_SPLIT = 16
GEMV_TARGET_CTAS = 512


def lt(key, dtype=None):
    with safe_open(os.path.join(MODEL, IDX[key]), framework="pt", device="cpu") as f:
        t = f.get_tensor(key)
    if dtype is not None:
        t = t.to(dtype)
    return t.to(DEV).contiguous()


def split_info(K, N):
    n_chunks = N // GEMV_N_TILE
    splits = max(1, (GEMV_TARGET_CTAS + n_chunks - 1) // n_chunks)
    splits = min(splits, K // (16 * GEMV_K_SPLIT))
    while (K // 16) % splits:
        splits += 1
    kt_max = (K // 16) // splits
    return n_chunks, splits, n_chunks * splits, kt_max


def profile_kernels(fn, iters=100):
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    agg = {}
    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.self_device_time_total > 0:
            agg[evt.key] = (evt.self_device_time_total / iters, evt.count)
    return agg


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
    mcg = base + ".mcg" in IDX and bool(lt(base + ".mcg").numel())
    mul1 = base + ".mul1" in IDX and bool(lt(base + ".mul1").numel())

    K = trellis.shape[0] * 16
    N = trellis.shape[1] * 16
    packed_bytes = trellis.numel() * 2
    tile_u16 = trellis.shape[2]
    Kbits = tile_u16 / 16 + (0.5 if tile_u16 % 16 else 0.0)

    x = (torch.randn(16, K, dtype=torch.float16, device=DEV) * 0.05).contiguous()
    A_had1 = torch.empty(1, K, dtype=torch.float16, device=DEV)
    A_had16 = torch.empty(16, K, dtype=torch.float16, device=DEV)
    C1 = torch.empty(1, N, dtype=torch.float16, device=DEV)
    C16 = torch.empty(16, N, dtype=torch.float16, device=DEV)
    buf = torch.empty_like(trellis)

    def gemv(M):
        A_had = A_had1 if M == 1 else A_had16
        C = C1 if M == 1 else C16
        def f():
            ext.exl3_gemm(x[:M], trellis, C, suh, A_had, svh, -1, mcg, mul1, 0)
        return f

    def ref_read():
        buf.copy_(trellis)

    ref = profile_kernels(ref_read, iters=50)
    ref_us = sum(v[0] for v in ref.values())

    print(f"\n== {name}  K={K} N={N} Kbits={Kbits:g} mcg={mcg} mul1={mul1}  packed={packed_bytes/1e6:.1f} MB ==")
    n_chunks, splits, ctas, kt_max = split_info(K, N)
    print(f"   split: n_chunks={n_chunks} splits={splits} CTAs={ctas} ({ctas/640:.1f} waves) kt_max={kt_max}")
    for M in (1, 16):
        agg = profile_kernels(gemv(M))
        had_us = sum(v[0] for k, v in agg.items() if "had" in k.lower() and "finalize" not in k)
        part_us = sum(v[0] for k, v in agg.items() if "partial" in k)
        fin_us = sum(v[0] for k, v in agg.items() if "finalize" in k)
        total_us = had_us + part_us + fin_us
        bw = packed_bytes / (part_us * 1e-6) / 1e12
        print(f"   M={M:<2} total={total_us:8.1f} us   had_in={had_us:7.1f}  partial={part_us:7.1f}  finalize={fin_us:7.1f}  "
              f"partial_BW={bw:5.2f} TB/s")
        for k, (us, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
            print(f"        {k[:78]:78s} {us:8.2f} us/call  n={n}")
    ref_bw = packed_bytes / (ref_us * 1e-6) / 1e12
    print(f"   ref copy: {ref_us:8.1f} us  ({ref_bw:.2f} TB/s read-side of a read+write copy)")
    del trellis, suh, svh, x, A_had1, A_had16, C1, C16, buf
    torch.cuda.empty_cache()

print("\ndone")
