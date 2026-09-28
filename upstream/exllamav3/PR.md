# exllamav3 — Fix EXL3 4-bit (k=4.0) dequant miscompile on ROCm/gfx1100

**Upstream:** the `exllamav3` extension (ROCm/HIP build)
**File:** `exllamav3_ext/quant/exl3_dq.cuh`, function `dq_dispatch`
**Patch:** [`0001-exllamav3-exl3-4bit-dequant-rocm-gfx1100.patch`](0001-exllamav3-exl3-4bit-dequant-rocm-gfx1100.patch) (single hunk, applies with `patch -p1`)

## Symptom

On AMD **gfx1100** (RDNA3, e.g. RX 7800/7900 XT), EXL3-quantized checkpoints whose
4-bit trellises use `bits == 4` (k=4.0) **dequantize to the wrong values**. The
observable effect on the OrcaSAQ-2-27B checkpoint is a top-1 regression:
`"The capital of France is"` generated `" the"` instead of `" Paris"`.

51 tensors are affected: all 16 full-attention `v_proj`, plus late-layer MLPs and
the MTP head.

## Root cause

`dq_dispatch` has a hand-optimized 4-bit path, `dq8_aligned_4bits<cb>`. That path
relies on the portable `FSHF_IMM` funnel-shift macro (the PTX `shf.r` replacement
in `ptx.cuh`). On ROCm/gfx1100 the compiler generates a funnel shift for
`FSHF_IMM` that extracts the **wrong 16-bit window**, so the first 8 columns of
every 16-column tile map to the wrong codebook entry.

The fault is isolated to `FSHF_IMM`: the K=2 path (`BFE16_IMM` + `fshift`, no
`FSHF_IMM`) dequantizes correctly on the same hardware, and the K=3 path already
routes through the generic `dq8<bits, cb, 4>` and is correct.

## Fix

Route `bits == 4` through the generic `dq8<bits, cb, 4>` path (align==4). It
computes the windows through `fshift` — the same 64-bit shift the working K=3
path uses — and is **bit-identical to CUDA** on this GPU.

```c++
else if constexpr (bits == 4)
{
    dq8<bits, cb, 4>(ptr, idx, frag0, frag1);   // was: dq8_aligned_4bits<cb>(...)
}
```

This applies to both `ext.reconstruct` and the serving `ext.exl3_gemm` (both call
the same reconstruct kernel).

## Verification

- **Raw dequant, bit-exact:** with the fix, the 4-bit reconstruct is
  bit-identical to the CUDA reference (`frac_wrong = 0`). Repro:
  `dbg2/reconstruct_raw.py` (CUDA + ROCm) and `dbg2/diff_raw.py`.
- **Full-model oracle:** the top-1 flips from `" the"` back to `" Paris"`
  (the correct token) on the OrcaSAQ-2-27B checkpoint, using the exact serving
  kernel `ext.exl3_gemm`.
- The K=2.0 and K=3.0 paths are unchanged.

## Scope / notes

- Developed and verified against the OrcaSAQ2 ROCm fork
  (`rocm/exllamav3_ext/quant/exl3_dq.cuh`). The change is a single, localized edit
  to `dq_dispatch`, which is shared with canonical exllamav3, so the hunk should
  apply to the upstream `exllamav3_ext/quant/exl3_dq.cuh` (verify surrounding
  context if your tree has diverged).
- The `dq8_aligned_4bits` path is left in place (untouched) so the CUDA
  optimized path is unaffected; this is purely a routing change for the 4-bit
  case. A follow-up could audit `FSHF_IMM` more broadly, but the minimal,
  verified fix is to route 4-bit through the known-good generic path.
