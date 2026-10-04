"""#14: interleaved same-process A/B of the staged GEMV partial kernel, matched full-path context.

V1 of this script compared a pure MODE 0 partial loop (old) against the full exl3_gemm path
(new: had + partial + finalize) -- not apples-to-apples; it left an unexplained +13% qkv
signal that could be the finalize/had context perturbing L2 between partial iters, not the
staged kernel itself. This version copies the production finalize kernel into the bench
(bench_finalize) so the old side can run in the exact production context:

  old_pure = MODE 0 partial, pure loop                          (v1 "old" arm; pure-loop reference)
  old_full = had_r_128 + MODE 0 partial + bench_finalize       (old production path, same buffers)
  new      = ext.exl3_gemm                                     (installed ext; staged partial for M <= 8)
  ref      = MODE 4 partial, pure loop                         (standalone staged kernel, pure-loop reference)

Interleaved per shape and M over ROUNDS. Per arm: the partial-kernel us (by kernel name) and
the total GPU us of the whole loop (had + partial + finalize where present). The
ship-or-revert metric is new_total vs old_full_total, plus the partial-vs-partial delta.

Run inside the dev container:  python3 /work/dbg2/gemv_ab.py [M ...]   (default M=1)
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from torch.profiler import ProfilerActivity, profile

from exllamav3.ext import exllamav3_ext as ext

sys.path.insert(0, os.environ.get("BENCH_BUILD_DIR", "/tmp/gemvbench"))
import gemv_bench

MODEL = os.environ.get("MODEL_PATH", "/model")
DEV = torch.device("cuda", 0)
IDX = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]


def lt(key, dtype=None):
    with safe_open(os.path.join(MODEL, IDX[key]), framework="pt", device="cpu") as f:
        t = f.get_tensor(key)
    if dtype is not None:
        t = t.to(dtype)
    return t.to(DEV).contiguous()


def arm_time(fn, want_substr, iters=100):
    """(partial_us, total_gpu_us) of looping fn(); the partial is identified by kernel name."""
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    part_us, tot_us = None, 0.0
    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.self_device_time_total > 0:
            tot_us += evt.self_device_time_total / iters
            if want_substr in evt.key:
                part_us = (part_us or 0.0) + evt.self_device_time_total / iters
    assert part_us is not None, f"kernel '{want_substr}' not found"
    return part_us, tot_us


SHAPES = {
    "qkv": "model.language_model.layers.0.linear_attn.in_proj_qkv",
    "gate": "model.language_model.layers.0.mlp.gate_proj",
    "down": "model.language_model.layers.0.mlp.down_proj",
    "q": "model.language_model.layers.3.self_attn.q_proj",
}

Ms = [int(x) for x in sys.argv[1:]] or [1]
ROUNDS = 3

for name, base in SHAPES.items():
    trellis = lt(base + ".trellis")
    suh = lt(base + ".suh", torch.float16)
    svh = lt(base + ".svh", torch.float16)
    mul1 = lt(base + ".mul1")
    K = trellis.shape[0] * 16
    N = trellis.shape[1] * 16
    for M in Ms:
        x = (torch.randn(M, K, dtype=torch.float16, device=DEV) * 0.05).contiguous()
        A2 = torch.empty_like(x)
        y = torch.empty(M, N, dtype=torch.float16, device=DEV)
        # M=16 production partial keeps the HBM-x kernel (gemv_partial_kernel<...>), not staged.
        new_substr = "staged" if M <= 8 else "partial_kernel<"

        def old_full():
            ext.had_r_128(x, A2, suh, None, 1.0)
            cp = gemv_bench.bench_partial(A2, trellis, M, 0)
            gemv_bench.bench_finalize(cp, y, svh)

        op_p, of_p, of_t, nw_p, nw_t, rf_p = [], [], [], [], [], []
        for _ in range(ROUNDS):
            a = arm_time(lambda: gemv_bench.bench_partial(A2, trellis, M, 0), "variant")
            op_p.append(a[0])
            a = arm_time(old_full, "variant")
            of_p.append(a[0]); of_t.append(a[1])
            a = arm_time(
                lambda: ext.exl3_gemm(x, trellis, y, suh, A2, svh, -1, False, bool(mul1.numel()), 0),
                new_substr)
            nw_p.append(a[0]); nw_t.append(a[1])
            a = arm_time(lambda: gemv_bench.bench_partial(A2, trellis, M, 4), "variant")
            rf_p.append(a[0])
        med = lambda v: sorted(v)[len(v) // 2]
        o_p, f_p, f_t, n_p, n_t, r_p = (
            med(op_p), med(of_p), med(of_t), med(nw_p), med(nw_t), med(rf_p))
        print(
            f"{name:5s} M={M:<2d} us  old_pure={o_p:7.1f}  "
            f"old_full p={f_p:7.1f} tot={f_t:7.1f}  "
            f"new p={n_p:7.1f} tot={n_t:7.1f}  "
            f"d_p={(n_p / f_p - 1) * 100:+5.1f}%  d_tot={(n_t / f_t - 1) * 100:+5.1f}%  "
            f"ref_pure_mode4={r_p:7.1f}"
        )
        del x, A2, y
    del trellis, suh, svh, mul1
    torch.cuda.empty_cache()

print("done")
