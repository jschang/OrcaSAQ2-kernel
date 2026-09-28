"""Per-op timing breakdown of the unfused EXL3 GEMM path on ROCm.

Stages (mirrors rocm_gemm.cu exl3_gemm):
  1. had_in     = had_r_128(x, A_had, suh, None, 1.0)      # input hadamard + suh
  2. dequant    = reconstruct(w, trellis, K, mcg, mul1)     # trellis -> (k,n) fp16 in HBM
  3. gemm       = hgemm_recon(A_had, w, C)                  # (m,k)@(k,n) via rocBLAS
  4. had_out    = had_r_128(C, C, None, svh, 1.0)           # output hadamard + svh

Times each at M=1,2,4,8 and reports the dequant HBM traffic (the term a fused GEMV
removes) so we can see exactly where the decode GEMM time goes.
"""
import json
import os
import torch
from safetensors import safe_open
from exllamav3.ext import exllamav3_ext as ext

MODEL = os.environ.get("MODEL_PATH", "/model")
DEV = torch.device("cuda", 0)
IDX = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]


def lt(key, dtype=None):
    with safe_open(os.path.join(MODEL, IDX[key]), framework="pt", device="cpu") as f:
        t = f.get_tensor(key)
    if dtype is not None:
        t = t.to(dtype)
    return t.to(DEV).contiguous()


def time_fn(fn, iters=200):
    # warmup
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


def breakdown(base, Ms=(1, 2, 4, 8), iters=200):
    trellis = lt(base + ".trellis")
    suh = lt(base + ".suh", torch.float16)
    svh = lt(base + ".svh", torch.float16)
    has_mcg = base + ".mcg" in IDX
    has_mul1 = base + ".mul1" in IDX
    mcg = bool(lt(base + ".mcg").numel()) if has_mcg else False
    mul1 = bool(lt(base + ".mul1").numel()) if has_mul1 else False
    K = trellis.shape[0] * 16
    N = trellis.shape[1] * 16
    K_bits = trellis.shape[2] / 16

    dequant_bytes = K * N * 2  # the (k,n) fp16 round-trip a fused GEMV removes
    label = base.split("layers.")[-1]
    print(f"\n== {label}  K={K} N={N} Kbits={K_bits:.1f} mcg={mcg} mul1={mul1}  dequant_HBM={dequant_bytes/1e6:.0f} MB ==")
    print(f"{'M':>3} {'total':>8} {'had_in':>8} {'dequant':>8} {'gemm':>8} {'had_out':>8} {'dq_BW':>9} {'gemm_BW':>9}")
    for M in Ms:
        x = torch.randn(M, K, dtype=torch.float16, device=DEV).contiguous()
        A_had = torch.empty_like(x)
        w = torch.empty((K, N), dtype=torch.float16, device=DEV)
        C = torch.empty(M, N, dtype=torch.float16, device=DEV)

        t_total = time_fn(lambda: ext.exl3_gemm(x, trellis, C, suh, A_had, svh, -1, mcg, mul1, 0), iters)
        t_had_in = time_fn(lambda: ext.had_r_128(x, A_had, suh, None, 1.0), iters)
        t_dequant = time_fn(lambda: ext.reconstruct(w, trellis, K_bits, mcg, mul1), iters)
        t_gemm = time_fn(lambda: ext.hgemm_recon(A_had, w, C), iters)
        t_had_out = time_fn(lambda: ext.had_r_128(C, C, None, svh, 1.0), iters)

        sum4 = t_had_in + t_dequant + t_gemm + t_had_out
        dq_bw = dequant_bytes / (t_dequant * 1e-3) / 1e12
        gemm_bw = (dequant_bytes + M * N * 2 + M * K * 2) / (t_gemm * 1e-3) / 1e12
        print(f"{M:>3} {t_total:>8.4f} {t_had_in:>8.4f} {t_dequant:>8.4f} {t_gemm:>8.4f} {t_had_out:>8.4f} "
              f"{dq_bw:>8.2f}T {gemm_bw:>8.2f}T   (sum4={sum4:.4f} vs total={t_total:.4f})")


b0 = "model.language_model.layers.0"
b3 = "model.language_model.layers.3"
breakdown(b0 + ".linear_attn.in_proj_qkv")   # K=5120 N=10240
breakdown(b0 + ".mlp.gate_proj")             # K=5120 N=17408 (big FFN)
breakdown(b0 + ".mlp.down_proj")             # K=17408 N=5120 (big K)
breakdown(b3 + ".self_attn.q_proj")          # K=5120 N=12288 (Kbits=3.5)
print("\ndone")
