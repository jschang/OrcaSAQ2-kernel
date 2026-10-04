# ROCm performance — candidate rabbit holes

Context: single-stream decode on the gfx1100 card does ~7 tok/s vs ~30 tok/s on the
NVIDIA card (64.5 tok/s measured on CUDA/16 GB, `presets/16gb.env`). The gap is ~4.3x.
Model: 64 layers, hidden 5120, 48 GDN layers + 16 full-attention layers (every 4th),
EXL3 trellis-quantized weights.

Status column: `open` = not yet touched, `verify` = needs measurement on the live
container before investing, `done` = implemented.

## Verification results (live container, 2026-10-01)

Container `blissful_sinoussi` (`orcasaq2-vllm-rocm:0.30.1`, vLLM v0.30.0+rocm723,
torch 2.12.0+git / hip 7.2.53211), stopped afterwards to free the GPU for GEMM work
(`docker start blissful_sinoussi` brings it back with the same env).

- **Hardware**: host has 6x 24 GB gfx1100 Navi 31 (device 0x744c, RX 7900 XTX class,
  ~1 TB/s HBM). The server ran on one of them (host idx 3 per in-container rocm-smi;
  container env has `HIP_VISIBLE_DEVICES=1`).
- **Launch**: `PRESET=16gb` + `MAXLEN=131072` override, `--gpu-memory-utilization 0.95`,
  `--max-num-seqs 4`, `--kv-cache-dtype fp8_e4m3`, `--block-size 32`,
  `--kv-offloading-size 8` (native), prefix caching on, `VLLM_SLEEP_WHEN_IDLE=1`.
  No `--enforce-eager`, no `--speculative-config`, no `--cpu-offload-gb`.
- **V1 (graphs) — RESOLVED: graphs are ON.** `enforce_eager=False`,
  `cudagraph_mode=FULL_AND_PIECEWISE`, capture sizes `[1,2,4,8]`,
  "Capturing CUDA graphs (PIECEWISE) 4/4", "(FULL) 2/2", graph pool 0.97 GiB.
  Not an eager-run; the 2.4x eager penalty does not apply.
- **V2 (GDN kernels) — CONFIRMED suboptimal path.**
  `rocm_aiter_ops.are_gdn_triton_kernels_available() = None`,
  `is_rdna_gdn_triton_kernels_available() = False`; container env
  `AITER_ROCM_ARCH=gfx942;gfx950` (gfx1100 not built). So all 48 GDN layers decode
  through the generic FLA Triton path
  (`fused_recurrent_gated_delta_rule_packed_decode` + causal-conv update) — keep as a
  rabbit hole, profile before investing.
- **Checkpoint bit-rates** (safetensors headers of /model; 26.02B trellis weights,
  avg **3.361 bits/weight**, 11.43 GiB on disk):

  | K (bits) | tensors | weight share |
  |---|---|---|
  | 2    | 6   | 2.1%  |
  | 3    | 231 | 57.4% |
  | 3.5  | 120 | 24.5% |
  | 4    | 51  | 11.2% |
  | 6    | 1   | 4.9%  |

  Consequences:
  - D1 (fast 4-bit dequant) is **deprioritized**: 4-bit is only 11.2% of the weights,
    and the generic `dq8` path it would replace also serves K=3 (57.4%), so the saving
    is a small delta on a small slice.
  - The generic-path funnel shifts (`fshift`, 64-bit) are the baseline for ~95% of
    weights; any per-weight dequant ALU win applies broadly, but the dequant ALU is
    not the bottleneck — the HBM traffic of materialization is.
- **Throughput observed in the running server's logs**: 6.3–6.5 tok/s single stream,
  8.7 tok/s with 2 concurrent reqs. KV pool 167,480 tokens (5.54 GiB), weights
  13.55 GiB, max model len 131,072.
- **Arithmetic check (confirms the diagnosis)**: quantized bytes = 26.02B w x
  3.361/8 = 10.9 GB. Current ROCm path moves ~0.42 (quant read) + 2 (fp16 write) +
  2 (fp16 read) = 4.42 bytes/weight per token = **~115 GB/token** → ~110 ms/token at
  ~1 TB/s → ceiling ~8–9 tok/s. Observed 6.3–8.7 tok/s sits exactly under that ceiling;
  the unfused GEMM is the binding constraint. A fused GEMV (10.9 GB/token) should
  lift the ceiling to ~60–90 tok/s before GDN/attention overheads are counted.

## P0 — verify first (V1/V2 done, see verification results; V3 open)

### V1. Are CUDA/HIP graphs actually being captured? `done — RESOLVED: graphs are on`
The plugin was first validated with `--enforce-eager` (see ROCM-DEBUG-2.md); eager
measured 2.4x slower on CUDA (`presets/16gb.env:21`). The preset does not set EAGER, so
graphs *should* be on, but the custom op `orcasaq2::shard_gemm` (`orcasaq2/ops.py:37`)
and the in-capture allocations inside `exl3_gemm` (the per-call `at::empty` fp16 weight,
`rocm/exllamav3_ext/rocm_gemm.cu:106`) are the two things that could silently force a
fallback.
Check: server startup log for graph-capture lines; `env | grep -i eager` in the
container; timing of a no-op request. If eager: just unsetting EAGER may be the whole
fix.

### V2. Which GDN decode kernels are active on RDNA3? `done — CONFIRMED: generic FLA Triton (no aiter RDNA kernels)`
48 of 64 layers are GDN. `third_party/vllm-rocm-debug/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:75-86`
gates the AITER Triton fast path on `are_gdn_triton_kernels_available() or
is_rdna_gdn_triton_kernels_available()`; otherwise the generic FLA Triton path
(fused recurrent update + causal_conv1d_update) runs. The recurrent update moves
~144 MB of SSM state per token across all GDN layers, so an unoptimized Triton
fallback is a real fixed cost.
Check: `python -c "from vllm._aiter_ops import rocm_aiter_ops; print(...)"` in the
container; rocprof a decode step and look for the GDN kernels' share.

### V3. Profile one decode step to get the actual split. `open`
rocprof (or at minimum `torch.profiler` with CUDA/HIP activities) on a single-stream
decode of a few hundred tokens. The arithmetic check in the verification results
already pins GEMM traffic at ~115 GB/token (ceiling ~8–9 tok/s vs 6.3 observed), so
GEMM is the expected dominant term; profile after the GPU is free to confirm and to
size the GDN/attention remainder for later work.

## P1 — the main fish: fused decode GEMM (est. ~3x on the GEMM share)

### G1. Replace the unfused exl3_gemm with a fused GEMV for small M. `done`
On ROCm, `exl3_gemm` (`rocm/exllamav3_ext/rocm_gemm.cu:60`) is a stand-in: input
Hadamard → `reconstruct` the **entire weight tile into a fresh fp16 matrix in HBM**
(line 106-107) → `at::mm`/rocBLAS (line 40) → output Hadamard. Per decode token every
weight is read quantized + written fp16 + read fp16 ≈ 4.42 bytes/weight at the pack's
avg 3.36 bits/weight (verification results), vs ~0.42 bytes/weight for a fused kernel
that streams the trellis once. At 10.9 GB of quantized layer weights that is
~115 GB HBM traffic per token → ~110 ms/token on the ~1 TB/s card, i.e. the ~8–9
tok/s ceiling the observed 6.3–8.7 tok/s sits under. Confirmed by the running
container's logs, not just arithmetic.

Plan:
- New HIP kernel for M ≤ ~8 (decode): dequant the packed trellis tile into
  **shared memory** — reuse `reconstruct_tile` (`rocm/exllamav3_ext/quant/reconstruct.cu:17`),
  already verified bit-identical to CUDA — writing to smem instead of HBM — then a
  WMMA (or scalar-FMA) dot of the tile against the (Hadamard-transformed) x rows.
  Output straight to C. No fp16 materialization, no rocBLAS.
- Dispatch: `orcasaq2/ops.py:46` already splits on `GEMM_MAX_ROWS` (144); route M ≤ 8
  to the GEMV kernel, 9–144 to the current path for now (rocBLAS hgemm is tolerable
  at mid M; revisit after profiling).
- Bonus: removes the per-call allocation chain (`at::empty` weight, `at::mm` out,
  `.to()`, `copy_` in `hgemm_recon` `rocm_gemm.cu:40-41`), which also inflates the
  graph memory pool.
- Verification: reuse `dbg2/gemm_test.py` (EXL3 GEMM vs reconstruct+dense) and the
  bit-exact dequant harness (`dbg2/diff_raw.py`); then end-to-end tok/s.

Result (done):
- Kernel: `rocm/exllamav3_ext/rocm_gemv.cu` — split-K `gemv_partial_kernel` (256 threads,
  128 cols/CTA, dequant a 16x128 k-tile into smem reusing `reconstruct_tile`'s
  dequant+shuffle, dot against x, fp32 partial) + `gemv_reduce_kernel` (sum splits -> fp16).
  Host `exl3_gemv` allocates the partial, launches partial via `hipLaunchKernel` (function
  pointer) + reduce via `<<<>>>`. 112 instances: K=1..8 x cb=0,1,2 x M=1,2,4,8 + K=1..3
  HALF=true. Dispatch in `exl3_gemm`: `!c_fp32 && M in {1,2,4,8} && N%128==0 && K%256==0`.
- Per-op M=1 (dbg2/gemm_breakdown.py, new GEMV vs old unfused): in_proj_qkv 0.277->0.137
  ms (2.0x), gate_proj 0.542->0.199 (2.7x), down_proj 0.528->0.201 (2.6x), q_proj
  0.373->0.143 (2.6x). M=8: 1.3-1.5x.
- Correctness: dbg2/gemm_test.py M=1/2/4/8 all shards cos=1.000000 (max err 0.002-0.019);
  M=5 fallback unchanged.
- End-to-end (single stream, 24 GB gfx1100, MAXLEN=8192 MAXSEQS=1, no KV offload):
  ~16.2-16.3 tok/s (runs 16.29/16.26/16.21) vs prior 6.3-8.7 unfused -> ~2.3x. Decode
  runs at M=1 (cudagraph_capture_sizes=[1,2]) so the GEMV path is the one active.
- Follow-up (ticket #10, done): `gemv_finalize` fuses the split-reduce and the output
  128-point Hadamard (+ svh) into one kernel (replaces reduce + had_r_128 and the fp16 C
  round trip); adaptive K_SPLIT (min(K/256, ceil(512/n_chunks)), walked to a divisor of
  K/16) instead of always K/256; M=16 dispatches to the GEMV (wins 3 of 4 representative
  shapes — gate -7.7%, down -7.1%, q -3.8%, qkv +3.5% — rows 9-144 stay unfused).
  Per-op A/B vs the pre-follow-up build: M=8 qkv -16.5%, gate -20.5%, down -21.6%,
  q -20.5% (adaptive splits); M=1 -1.4 to -7.0% (epilogue). Server (16 GB, single seq,
  matched ~252-token prompt): GEMV family 48.71 -> 47.57 ms/tok, host gap 6.20 -> 4.95
  ms/tok (404 fewer launches/step), wall 62.80 -> 61.48 ms/tok (15.9 -> 16.3 tok/s
  profiled). Correctness cos=1.000000 all shapes.
- Ticket #14 (done): the 0.71 TB/s figure above was wrong — the partial kernel runs at
  0.07-0.21 TB/s effective (dbg2/gemv_bw.py): latency/ALU-bound (~2.3-3.2 us/tile vs ~1 us
  HBM-only), not bandwidth-bound. A mode-matrix bench (dbg2/gemv_bench.py, MODE 0-4; MODE 0
  and the staged variant verified ISA-identical to the production kernels by disassembly)
  splits the M=1 qkv cost: dequant ALU ~40%, FMA ~40%, x read/staging ~20%. Shipped
  `gemv_partial_staged_kernel` (rocm_gemv.cu): the x slice is staged in smem once per tile
  and the FMA runs as v_dot2_f32_f16 column pairs for M in {1,2,4,8} (M=16 keeps the HBM-x
  path; the stage crushes occupancy at 16 rows). Per-op two-build A/B: 0..-3.1% (M=1, the
  x slice is only ~20% of the cost); E2E +2.5% (17.03 -> 17.46 tok/s on rocm-1, 252-token
  decode, non-overlapping runs, drift-checked). Measurement caveat: standalone per-kernel
  bench on this box is session-drift-dominated — ISA-identical kernels drift +/-25%
  between sessions and the staged/old ratio flips direction — so only same-context
  production A/B and E2E count. Next lever: the dequant ALU (~40% of GEMV time): wider
  dequant / packed-FMA. Attention + GDN remain the next largest terms per the #7 profile.

## P2 — deferred (deprioritized by the checkpoint histogram)

### D1. Restore a fast 4-bit dequant path instead of the generic fallback. `deferred`
Histogram (verification results): 4-bit is only 11.2% of the weights, and the generic
path this would replace already serves K=3 (57.4%) at what is evidently a tolerable
speed. Saving is a small delta on a small slice. Keep as a follow-up if dequant ALU
ever shows up in a profile.
`rocm/exllamav3_ext/quant/exl3_dq.cuh:306-316`: the gfx1100 miscompile fix reroutes
4-bit rates through generic `dq8<bits,cb,4>` (line 315). The generic path does 8 funnel
shifts, each a 64-bit merge + shift (`fshift`, line 7; `FSHF_IMM`, `ptx.cuh:30`) and
RDNA3 has no 64-bit ALU, so each compiles to a multi-instruction sequence. The original
`dq8_aligned_4bits` (line 163) needed only one 20-bit funnel shift, which has a trivially
safe 32-bit-only form: `s = (b << 12) | (a >> 20)` — constant shift + OR, no 64-bit
temp, nothing for the compiler to miscompile.
If ever picked up: ~10-line change plus the existing bit-exact tests
(`dbg2/diff_raw.py`, `dbg2/compare_dequant.py`).

## P3 — config-level, no code (est. 1.5–1.7x if it applies)

### M1. MTP speculative decoding. `open`
`presets/16gb-mtp.env` measured 1.72x on CUDA (64.5 → 111.2 tok/s) with
`EXTRA='--speculative-config {"method":"qwen3_next_mtp","num_speculative_tokens":2}'`.
Requires the -mtp checkpoint variant (the running container's /model is the plain
variant); the draft head takes KV from the same pool, which is already tight
(167,480-token pool at MAXLEN=131072). Cheap to test if/when the -mtp pack is
available: one env line.

## P4 — lower priority / subsumed

- rocBLAS skinny-M hgemm (M 9–144): revisit only if the mid-range batch profile shows it
  matters; subsumed by G1 for decode.
- The two Hadamard launches per shard per token (`rocm_gemm.cu:103,113`): free under
  graph capture; only costs anything if V1 lands on eager.
- KV offload over PCIe (`KV_OFFLOAD_GB=8`, native backend): only hurts on prefix-cache
  misses; not expected to matter single-stream.
- `xf = x.half()` / `out.to(x.dtype)` per layer (`orcasaq2/config.py:233,243`): tiny
  activation-size tensors, negligible.

## Notes

- Everything in P1/P2 lives in `rocm/exllamav3_ext/`; the dispatch hook is
  `orcasaq2/ops.py`. The CUDA reference kernels are in the nested clone
  `upstream/exllamav3/exllamav3/exllamav3/exllamav3_ext/quant/` (fused cooperative-group
  path — porting it whole is the heavy option; the smem-dequant GEMV above is the light one).
- Measured reference points: CUDA 16 GB 64.5 tok/s (no MTP), 111.2 tok/s (MTP);
  ROCm 24 GB gfx1100 ~6.3–8.7 tok/s (current, verification results).
