"""Large-M (prefill) sweep of the unfused EXL3 GEMM path on ROCm. Companion to
gemm_breakdown.py (M=1..8). Mirrors the production dispatch in rocm_gemm.cu /
orcasaq2/ops.py:

  M <= 144 (GEMM_MAX_ROWS): ext.exl3_gemm  ->  M in {1,2,4,8,16}: fused GEMV
                            else: had_in + reconstruct + hgemm_recon + had_out
  M >  144:                   python does exactly had_in + reconstruct +
                              hgemm_recon + had_out   (== the exl3_gemm fallback
                              branch, so we measure the 4 ops directly)

Runs one representative linear per unique (K,N) shape, M in (128, 512, 2048).
"""
import json
import os
import torch
from safetensors import safe_open
from exllamav3.ext import exllamav3_ext as ext

MODEL = os.environ.get("MODEL_PATH", "/model")
DEV = torch.device("cuda", 0)
IDX = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
Ms = (128, 512, 2048)


def lt(key, dtype=None):
    with safe_open(os.path.join(MODEL, IDX[key]), framework="pt", device="cpu") as f:
        t = f.get_tensor(key)
    if dtype is not None:
        t = t.to(dtype)
    return t.to(DEV).contiguous()


def time_fn(fn, iters):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


# one representative trellis per unique (K,N), smallest N first (cheapest)
shapes = {}
for key in IDX:
    if key.endswith(".trellis"):
        with safe_open(os.path.join(MODEL, IDX[key]), framework="pt", device="cpu") as f:
            t = f.get_slice(key)
        K, N, _ = (int(t.get_shape()[0]) * 16, int(t.get_shape()[1]) * 16, int(t.get_shape()[2]))
        if (K, N) not in shapes:
            shapes[(K, N)] = key

print(f"unique (K,N) shapes: {sorted(shapes)}")
for (K, N), trellis_key in sorted(shapes.items()):
    base = trellis_key[:-len(".trellis")]
    if N > 20000:
        print(f"\n== {base.split('layers.')[-1]}  K={K} N={N}  SKIPPED (lm_head-sized; "
              f"never run at large M in production — logits are one row) ==")
        continue
    trellis = lt(trellis_key)
    suh = lt(base + ".suh", torch.float16)
    svh = lt(base + ".svh", torch.float16)
    has_mcg = base + ".mcg" in IDX
    has_mul1 = base + ".mul1" in IDX
    mcg = bool(lt(base + ".mcg").numel()) if has_mcg else False
    mul1 = bool(lt(base + ".mul1").numel()) if has_mul1 else False
    K_bits = trellis.shape[2] / 16
    quant_bytes = trellis.numel() * 2
    label = base.split("layers.")[-1]
    dequant_bytes = K * N * 2
    print(f"\n== {label}  K={K} N={N} Kbits={K_bits:.1f} mcg={mcg} mul1={mul1}  "
          f"quant={quant_bytes/1e6:.1f} MB  dequant={dequant_bytes/1e6:.1f} MB ==")
    print(f"{'M':>5} {'had_in':>9} {'dequant':>9} {'gemm':>9} {'had_out':>9} {'sum4':>9} "
          f"{'dq_GB/s':>9} {'gmm_GB/s':>9} {'gmm_TFLOP':>10}")
    for M in Ms:
        iters = 200 if M <= 512 else 100
        x = torch.randn(M, K, dtype=torch.float16, device=DEV).contiguous()
        A_had = torch.empty_like(x)
        w = torch.empty((K, N), dtype=torch.float16, device=DEV)
        C = torch.empty(M, N, dtype=torch.float16, device=DEV)

        t_had_in = time_fn(lambda: ext.had_r_128(x, A_had, suh, None, 1.0), iters)
        t_dequant = time_fn(lambda: ext.reconstruct(w, trellis, K_bits, mcg, mul1), iters)
        t_gemm = time_fn(lambda: ext.hgemm_recon(A_had, w, C), iters)
        t_had_out = time_fn(lambda: ext.had_r_128(C, C, None, svh, 1.0), iters)

        s4 = t_had_in + t_dequant + t_gemm + t_had_out
        gmm_bytes = dequant_bytes + M * N * 2 + M * K * 2
        gmm_flops = 2.0 * M * K * N
        print(f"{M:>5} {t_had_in:>9.4f} {t_dequant:>9.4f} {t_gemm:>9.4f} {t_had_out:>9.4f} "
              f"{s4:>9.4f} {dequant_bytes/(t_dequant*1e-3)/1e12:>9.2f} "
              f"{gmm_bytes/(t_gemm*1e-3)/1e12:>9.2f} {gmm_flops/(t_gemm*1e-3)/1e12:>10.2f}")
        del x, A_had, w, C
        torch.cuda.empty_cache()

    del trellis, suh, svh
    torch.cuda.empty_cache()

print("\ndone")
