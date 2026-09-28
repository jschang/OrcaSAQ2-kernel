# OrcaSAQ2 ROCm debug — session 2 handoff

This is the handoff for **session 2** of the ROCm/vLLM EXL3 diagnosis, written
so another model/agent (e.g. gpt-5.5) can pick up exactly where this one left
off. Read `ROCM-DEBUG.md` and `ROCM-GPT-DEBUG.md` first for the earlier
chronology; this file is the incremental state after session 2.

All session-2 scripts and data live in **`dbg2/`** (kept separate from the
session-1 files in `dbg/`).

## Symptom (unchanged)

Serve the EXL3 OrcaSAQ2/Qwen3.5 hybrid on ROCm/gfx1100. Fresh single prompt:

```text
prompt: "The capital of France is"
max_tokens: 1
temperature: 0

CUDA/reference: top token " Paris", logprob ≈ -0.4135
ROCm target:    top token " the",   logprob ≈ -1.27   (in-process: -1.2283, " Paris" rank 10 @ -3.9158)
```

## What session 2 did: a full-model pure-torch oracle

The session-1 notes (`ROCM-GPT-DEBUG.md`) concluded the bug is in the
integrated vLLM ROCm execution path and proposed dumping hidden states per
layer. Session 2 took the strongest possible form of that: a **from-scratch
pure-torch oracle** of the whole model that uses the *exact* served EXL3
weights (same `exllamav3` GEMM kernel as serving) but does *every* non-GEMM op
in plain torch (fp32 conv/l2norm/gating, naive per-token GDN recurrence, bf16
SDPA, GemmaRMSNorm in fp32). If the oracle reproduces the CUDA answer, the
weights+math are fine and the runner is at fault; if it reproduces the ROCm
answer, the weights/dequant are at fault.

Key files (all in `dbg2/`):

```text
orca_oracle.py        the full-model oracle (the centerpiece)
orca_server_dump.py   in-process vLLM run that dumps per-layer states (ROCm server's answer)
compare_dumps.py      layer-by-layer oracle-vs-server comparison (CPU, no GPU)
gemm_test.py          EXL3 GEMM vs reconstruct_weight+dense over 7 layers
dequant_stats.py      statistics of reconstruct_weight output (magnitude sanity)
gdn_unit_test.py      unit test proving the two GDN transposition bugs
convtest.py           causal-conv tap-ordering check
inspect_dumps.py      dump structure inspector
cuda_export_weights.py  dequantize projs with the CUDA ext.reconstruct (run in dreamy_bassi)
compare_dequant.py      ROCm ext.reconstruct vs the CUDA reference (the cd=64 bug finder)
cuda_dequant_ref*.pt    CUDA FP16 dequant references (gitignored, large; regenerate as needed)
orca_debug.py         stage-by-stage isolation helper (imports orca_oracle)
qwen3_next_patched.py vLLM qwen3_next.py with layer-dump + rope instrumentation
sitecustomize.py      engine-side hooks (per-layer dump + rope-fix), alt dump path
img_orcasaq2_init.py  the IMAGE's orcasaq2/__init__.py (no mrope patch)
img_qwen3_next.py     the IMAGE's qwen3_next.py
stock_qwen3_next.py   stock vLLM qwen3_next.py
brain_qwen3_next.py   the CUDA brain container's qwen3_next.py (reference)
```

### Run commands (ROCm idle GPU, image `orcasaq2-vllm-rocm:0.30.0`)

The model is at `/mnt/sdd2/Models/orcarouter/OrcaSAQ-2-27B` on the host. Mount
`dbg2/` as `/dbg2` (all the scripts' default output paths assume this).

```bash
REPO=~/3code/OrcaSAQ2-kernel
MODEL=/mnt/sdd2/Models/orcarouter/OrcaSAQ-2-27B

# 1) Oracle (EXL3 GEMM path, ~a few min). Prints TOP-K + saves /dbg2/oracle_layer_states.pt
docker run --rm --device /dev/kfd --device /dev/dri \
  -v $MODEL:/model -v $REPO/dbg2:/dbg2 \
  -e HIP_VISIBLE_DEVICES=0 -e MODEL_PATH=/model -e OUT_DIR=/dbg2 \
  -e ORACLE_DENSE=0 -e ORACLE_SAVE_INT=1 \
  --entrypoint python orcasaq2-vllm-rocm:0.30.0 /dbg2/orca_oracle.py

# 2) GEMM consistency check (EXL3 kernel vs reconstruct_weight)
docker run --rm --device /dev/kfd --device /dev/dri \
  -v $MODEL:/model -v $REPO/dbg2:/dbg2 -e HIP_VISIBLE_DEVICES=0 \
  --entrypoint python orcasaq2-vllm-rocm:0.30.0 /dbg2/gemm_test.py

# 3) Dequant weight statistics
docker run --rm --device /dev/kfd --device /dev/dri \
  -v $MODEL:/model -v $REPO/dbg2:/dbg2 -e HIP_VISIBLE_DEVICES=0 \
  --entrypoint python orcasaq2-vllm-rocm:0.30.0 /dbg2/dequant_stats.py

# 4) Compare the two dumps (CPU only, no GPU)
docker run --rm -v $REPO/dbg2:/dbg2 \
  --entrypoint python orcasaq2-vllm-rocm:0.30.0 /dbg2/compare_dumps.py
```

`ORACLE_DENSE=1` makes the oracle dequantize every shard to dense bf16 up front
and use a plain torch GEMM instead of the `exll3_gemm` kernel — but the full
27B dense set does not fit in a single ~25.75 GB GPU, so it OOMs past layer ~4.
Use `ORACLE_DENSE=0` (the EXL3-kernel path) for full runs.

For the **server dump** (`orca_server_dump.py`) you additionally mount the
instrumented model + sitecustomize over the image copies:

```bash
QW=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py
docker run --rm --device /dev/kfd --device /dev/dri --shm-size=16g \
  -v $MODEL:/model -v $REPO/dbg2:/dbg2 \
  -v $REPO/dbg2/qwen3_next_patched.py:$QW \
  -v $REPO/dbg2/sitecustomize.py:/usr/local/lib/python3.12/dist-packages/sitecustomize.py \
  -e HIP_VISIBLE_DEVICES=0 -e MODEL_PATH=/model \
  -e LAYER_DUMP_OUT=/dbg2/server_layer_states.pt \
  -e ROPE_FIX_LOG=/dbg2/rope_fix.log \
  --entrypoint python orcasaq2-vllm-rocm:0.30.0 /dbg2/orca_server_dump.py
```

(Confirm the container's `sitecustomize` path with
`python -c "import sitecustomize, inspect; print(inspect.getfile(sitecustomize))"`
if the mount above is not the one that gets loaded.)

## Findings

### 1. The oracle's own GDN had three bugs (now fixed)

The oracle initially produced garbage ("atz" -4.2, worse than the server). The
GDN (linear-attention) layer in `orca_oracle.py` was wrong in three places, all
now fixed:

- **State-update outer product** (was element-wise, rank collapse):
  `S += kf[t] ⊗ v_new[t]` over the K×V axes, not `v_new.unsqueeze(2)*kf.unsqueeze(-1)`.
- **Output contraction** (wrong axis): `o[t] = einsum("hkv,hk->hv", S, qf[t])`,
  not `"hkv,hv->hk"`.
- **Causal-conv tap order (reversed)**: vLLM `causal_conv1d_fn` applies weight
  tap `j` to `x[t-(W-1-j)]` (last tap = current token). The oracle loop had
  `w[0]` on the current token. Fixed to `offset = CONV_W-1-j`.

These were "silent" only because `K_DIM == V_DIM == 128` in this model. The
correct recurrence matches the aiter triton kernel the server actually uses
(`aiter/.../gated_delta_rule/decode/fused_recurrent.py`). `gdn_unit_test.py`
shows the buggy version fails when K≠V and diverges (max|diff| ~4.9 out / 4.0
state) even with K==V.

### 2. After the fixes, the oracle MATCHES the ROCm server

With the GDN fixed, the oracle now produces **`" the" (-1.30)`** — the same top
token as the ROCm server (`" the" -1.27`), instead of the CUDA reference
(`" Paris"`). `compare_dumps.py` shows the two agree layer-by-layer:
cos ≈ 1.0 at layers 0–2, degrading to ≈ 0.78 by layer 63 (accumulating bf16
rounding between the pure-torch oracle and the ROCm kernels).

**Interpretation:** the independent pure-torch oracle agrees with the ROCm
server, so the non-GEMM math (GDN, attention, norms, MLP, residual glue) is
essentially correct on ROCm. The earlier "garbage" was entirely the oracle's
own GDN bugs, not the served model.

### 3. The EXL3 GEMM is consistent with the reference dequant

`gemm_test.py` compares the fused `ext.exl3_gemm` kernel against
`reconstruct_weight` + dense GEMM over 7 layers (in_proj_qkv, in_proj_z,
out_proj, q_proj, k_proj, gate_proj, down_proj):

```text
cos ≈ 1.000000, max_abs ≈ 0.004–0.012  (bf16 rounding)  on every layer
```

So the GEMM kernel itself is not the problem in isolation.

### 4. Dequantized weights have plausible statistics

`dequant_stats.py` on `reconstruct_weight` output:

```text
in_proj_qkv  std 0.0157  max 0.43
gate_proj    std 0.0102  max 0.42
q_proj       std 0.0172  max 0.32
```

Reasonable magnitudes for a 27B model — so the dequant is not *obviously*
broken, but plausible-looking weights can still be subtly wrong (sign /
permutation / index).

### 5. CUDA→ROCm dequant diff — the bug is `ext.reconstruct` at code_dim=64 (k=4.0)

The decisive experiment. `cuda_export_weights.py` runs INSIDE the CUDA reference
container (`dreamy_bassi`, image `orcasaq2-vllm:0.30.0`, model at `/model`) and
dequantizes a set of projections with the **CUDA** `ext.reconstruct`, saving FP16
to a `.pt`. `compare_dequant.py` runs the *same* pipeline on **ROCm** and diffs
against the CUDA reference. Both load the identical EXL3 tensors from the
checkpoint; the only difference is the `ext.reconstruct` kernel (CUDA vs ROCm).

Result across 12 projections spanning every trellis rate:

```text
projection (layer)   code_dim  k     CUDA == ROCm?
in_proj_qkv (0)        48     3.0    identical
out_proj (0)           48     3.0    identical
gate_proj (0)          56     3.5    identical
up_proj (0)            48     3.0    identical
down_proj (0)          48     3.0    identical
q_proj (3)             56     3.5    identical
k_proj (3)             56     3.5    identical
v_proj (3)             64     4.0    WRONG  (cos 0.644, ~0.3% elems match)
o_proj (3)             48     3.0    identical
v_proj (7)             64     4.0    WRONG  (cos 0.645)
gate_proj (53)         64     4.0    WRONG  (cos 0.655)  <- non-v_proj, still wrong
gate_proj (12)         32     2.0    identical
```

"identical" = `max_abs=0`, `exact4=1.0` (bit-for-bit after fp16 rounding).
**Every code_dim=64 (k=4.0) tensor is wrong** (cos ≈ 0.64–0.65, ~0.3% of elements
match, every row and every column affected); every other rate is bit-identical.

The trellis `code_dim` = 16×k (the EXL3 bitrate). The checkpoint uses five rates:

```text
code_dim=32 (k=2.0)    6 tensors   -> OK
code_dim=48 (k=3.0)  231 tensors   -> OK
code_dim=56 (k=3.5)  120 tensors   -> OK
code_dim=64 (k=4.0)   51 tensors   -> WRONG on ROCm
code_dim=96 (k=6.0)    1 tensor    -> lm_head, untested (see below)
```

So the **ROCm `ext.reconstruct` kernel is wrong specifically for code_dim=64
(k=4.0) trellises.** That is 51 tensors, including all 16 full-attention `v_proj`
(layers 3,7,…,63) and the gate/up/down of the later layers (53–63) plus the MTP
module — a large fraction of the model. It is rate-specific, not `v_proj`-specific:
layer-53 `gate_proj` (also code_dim=64) is wrong, while layer-12 `gate_proj`
(code_dim=32) is fine.

The bug is in `ext.reconstruct` (the core trellis dequant), **not** the Hadamard /
suh/svh scaling (pure torch, identical on both GPUs) and **not** the loader (the
other 11 projections match exactly through the same pipeline). This is the root
cause of the `" the"` vs `" Paris"` gap: the dequantized code_dim=64 weights are
wrong on ROCm, and the error accumulates through the layers.

`reconstruct_weight` (in `orcasaq2/__init__.py`), for reference:

```python
ext.reconstruct(w, trellis, k, has_mcg, has_mul1)   # core trellis dequant (exllamav3 kernel)  <-- WRONG for code_dim=64 on ROCm
w = _t128(w.t()).t() * suh.unsqueeze(1)             # row Hadamard + input scale   (pure torch, OK)
w = _t128(w)      * svh.unsqueeze(0)                # col Hadamard + output scale  (pure torch, OK)
return w.t().contiguous().half()
```

Run the experiment:

```bash
# CUDA side (inside dreamy_bassi; memory-frugal, one projection at a time)
docker exec dreamy_bassi env MODEL_PATH=/model OUT=/tmp/dequant_ref.pt \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /usr/bin/python3 /tmp/cuda_export_weights.py          # PROJECTIONS=... to override
docker cp dreamy_bassi:/tmp/dequant_ref.pt <repo>/dbg2/cuda_dequant_ref.pt

# ROCm side
docker run --rm --device /dev/kfd --device /dev/dri \
  -v <model>:/model -v <repo>/dbg2:/dbg2 -e HIP_VISIBLE_DEVICES=0 \
  -e MODEL_PATH=/model -e REF=/dbg2/cuda_dequant_ref.pt \
  --entrypoint python orcasaq2-vllm-rocm:0.30.0 /dbg2/compare_dequant.py
```

### Open item: lm_head (code_dim=96, k=6.0)

The single code_dim=96 tensor is `lm_head.trellis` (shape `[320,15520,96]`, ~950
MB), the final projection to logits. It could not be exported from
`dreamy_bassi`: the 4090 has only ~1.2 GB free and the trellis alone is ~950 MB
(a fresh process's CUDA context leaves < 1 GB). It needs either more headroom on
the 4090, a **chunked** export (split the trellis along the `oc/16` axis, e.g.
`[320,1000,96]` chunks, and reconstruct column-blocks), or a separate GPU. Given
the code_dim=64 bug is already confirmed, checking whether code_dim=96 is also
affected is the main remaining dequant question.

## Root cause, fix, and verification (session 3)

**Fixed.** The ROCm `ext.reconstruct` bug for code_dim=64 (k=4.0) is resolved: the
dequant now matches the CUDA reference bit-for-bit, and the full-model oracle
produces the correct top token.

### Root cause

The EXL3 4-bit dequant (`dq_dispatch` in `rocm/exllamav3_ext/quant/exl3_dq.cuh`)
routed `k=4.0` through a hand-optimized kernel `dq8_aligned_4bits`. That function
is byte-identical to upstream exllamav3 *except* for the bit-extraction helpers:
upstream uses PTX inline asm (`shf.r.wrap.b32` / `bfe.u32`), while the ROCm port
replaced them with portable C++ macros in `rocm/exllamav3_ext/ptx.cuh`:

```cpp
#define FSHF_IMM(dst, lo, hi, imm) (dst) = (uint32_t)((((uint64_t)(hi)) << 32) | (uint64_t)(lo)) >> (imm)
#define BFE16_IMM(dst, src, imm)  (dst) = ((src) >> (imm)) & 0xffffu
```

The macros are semantically correct, but on ROCm/gfx1100 the hand-optimized 4-bit
path miscompiles: the `FSHF_IMM` funnel shift extracts the wrong 16-bit windows, so
the first 8 columns of every 16-column tile are dequantized to the wrong codebook
value (~37% of the tensor wrong, the `(col % 16) < 8` pattern, with the wrong values
collapsing onto a few repeated codebook entries). The other rates are unaffected --
`k=3.0` uses the generic `dq8<bits,cb,4>` path (only `fshift`, which compiles
correctly), and the `k=2.0` path uses `BFE16_IMM` + `fshift` but *not* `FSHF_IMM`
(its 6 tensors in this checkpoint dequantize correctly) -- so the fault localizes
to the portable `FSHF_IMM`.

### Fix

Route the 4-bit rate through the generic `dq8<bits,cb,4>` path instead of the
hand-optimized one. The two compute **bit-identical** windows (same word access
`a=ptr[31-lane]`, `b=ptr[lane]`, same 8 windows); the generic path just derives
them via `fshift` + plain shifts — the same formulation the working `k=3.0` path
uses — and it matches CUDA exactly. One-line change in `dq_dispatch`
(`exl3_dq.cuh:306`), with a comment explaining why:

```cpp
    else if constexpr (bits == 4)
    {
        dq8<bits, cb, 4>(ptr, idx, frag0, frag1);
    }
```

This fixes both `ext.reconstruct` and the serving `ext.exl3_gemm`, since the GEMM
unpacks its weights through the same `reconstruct` kernel
(`rocm_gemm.cu:107` -> `reconstruct_kernel` -> `dq_dispatch`). A portable
`bfe64`-based 4-bit path (`dq8_aligned_4bits_bfe64`) also exists and *may* be
correct (it uses the `bfe64` function rather than the macros), but it was not the
one verified here.

### Verification

Rebuilt the ROCm ext (`PYTORCH_ROCM_ARCH=gfx1100`, `rocm/setup.py`) and re-ran:

- **Raw dequant diff** (`dbg2/reconstruct_raw.py` + `dbg2/diff_raw.py` on
  `layers.3.self_attn.v_proj`): CUDA ref vs fixed ROCm -> `cos=1.0`, `max_abs=0.0`,
  `frac_wrong=0.000000` (0/5242880), bit-identical (was `frac_wrong=0.3667`
  before the fix).
- **Full-model oracle** (`dbg2/orca_oracle.py`): top-1 token now `" Paris"` @
  `-0.4766` (was `" the"` @ ~-1.23). `" Paris"` matches the CUDA reference
  (`-0.4135`); `" the"` drops to rank 6. The small logprob gap vs the CUDA server
  is expected (the oracle runs all non-GEMM ops in pure torch).

### Remaining

- `lm_head` (code_dim=96, k=6.0) was never exported from the 4090 (OOM); it uses
  the `k=6.0` path (generic `dq4`), which was already correct, so it is expected
  to be fine — a chunked-export sanity check is worthwhile if headroom allows.
- **Do not re-chase** the non-GEMM math (GDN/attention/norms/MLP) or the
  `exl3_gemm` kernel — session 2 exonerates them (oracle-vs-server agreement;
  `gemm_test.py` cos≈1.0).

## Server startup (session 4) — real vLLM server now serves " Paris"

**Milestone:** the actual vLLM `serve-exl3` server (with the session-3 dequant fix
rebuilt into the ext) now boots and serves the model, returning the correct top
token:

```text
prompt: "The capital of France is"   max_tokens: 1   temperature: 0
GENERATED: " Paris"   logprob -0.405
top-5: " Paris" -0.405, " not" -3.405, " in" -3.655, " a" -3.717, " ______" -3.967
```

The dequant fix is thus confirmed through the *full* serving path (all 64 layers,
real attention/GDN/norm/MLP, the `exl3_gemm` kernel), not just the isolated
oracle. The logprob gap vs the oracle (`-0.4766`) and the CUDA server (`-0.4135`)
is within expected serving-path rounding.

Booting the server on this box required **four separate vLLM/container fixes, none
of which are part of the exllamav3 dequant fix** (committed in `b81c739`). They are
pre-existing vLLM-0.30.0-on-gfx1100 issues, and together they explain why an
earlier `serve-exl3` attempt froze the box.

### A. RoPE crash in the profile/dummy run (the real code bug)

The engine core died during memory profiling:

```text
rotary_embedding, pos_encoding_kernels.hip:141,
query, key and positions must have the same batch_size and seq_len
```

Root cause (Qwen3.5 / `qwen3_next.py`): the profile run hands the full-attention
layer a **2D** `positions` of shape `[num_reqs, seq_len]` (observed `[3, 2048]`)
while `q`/`k` are flattened `[num_tokens, hidden]` (observed `[2048, 6144]` /
`[2048, 1024]`). In `_project_qkv_gate` there are two paths:

- the **fused** path (`use_fused_qk_norm_rope_gate`) handles 2D positions
  correctly: `if positions.ndim == 2 and not mrope_section: positions = positions[0]`
  (→ `[2048]`, matching `q.size(0)`);
- the **non-fused** path (the one this model uses) does **not** apply that
  handling, so it passes the 2D `positions` straight to `rotary_emb`. The HIP RoPE
  kernel rejects it: the 1D branch fails (`q.size(0)==positions.size(0)` → `2048
  != 6144` after a naive flatten) and the 2D branch fails (`q.size(0)==positions
  .size(0)` → `2048 != 3`).

The **real fix belongs in vLLM**: the non-fused path should apply the same
`positions = positions[0]` as the fused path. For serving, a minimal monkey-patch
at the RoPE entry point is sufficient — apply the identical handling in
`RotaryEmbedding.forward_cuda`:

```python
def forward_cuda(self, positions, query, key=None):
    if positions.dim() == 2 and not getattr(self, "mrope_section", None):
        positions = positions[0]
    return orig(self, positions, query, key)
```

Installed in-container via a startup `.pth` (so the engine-core subprocess gets
it): `exl3_rope_patch.py` + `exl3_rope_patch.pth` (`import exl3_rope_patch`) in
`/usr/local/lib/python3.12/dist-packages/`. It only touches the 2D (profile-run)
case; normal serving uses 1D `positions` and is unaffected.

### B. `/dev/shm` size

The engine core needs ~8.2 GiB of `/dev/shm` (it mmaps an 8.58 GB offload
region); Docker's default 64 MiB is far too small:

```text
RuntimeError: Insufficient space in /dev/shm: 8183 MiB required, 64 MiB free.
```

Start the container with `--shm-size=16g` (or `--ipc=host`).

### C. tvm_ffi one-time build (the likely cause of the freeze)

The first `import vllm`/`import torch` in a *fresh* container triggers a one-time
build of the **optional** torch-C-dlpack addon
(`tvm_ffi/utils/_build_optional_torch_c_dlpack.py`), which forks **hundreds of
parallel** compile processes (`libtorch_c_dlpack_addon_torch212-rocm.so`) and
takes >10 min. Repeatedly triggering/abandoning it left 139+ orphaned processes —
the likely cause of the box freeze. Set `TVM_FFI_DISABLE_TORCH_C_DLPACK=1`
*container-wide* (checked at `tvm_ffi/_optional_torch_c_dlpack.py:206`; skips the
build when set) — `import torch`/`import vllm` then takes ~1 min and no runaway
processes appear.

### D. GPU readiness after prior runs

The first relaunch failed with `RuntimeError: No CUDA GPUs are available` because
the GPU was still recovering from the earlier model loads; a bare
`torch.cuda.device_count()` returned 1 ~2 min later and the relaunch then worked.
Re-check `rocm-smi` / `torch.cuda.device_count()` before a relaunch if the
previous run was heavy.

### Launch recipe (verified)

Container flags: `--device /dev/kfd --device /dev/dri --group-add video
--shm-size=16g`; mounts `rocm/` → `/opt/exl3-rocm-src` (ro), `dbg2/` → `/dbg2`
(rw), model → `/model` (ro); env `HIP_VISIBLE_DEVICES=0`,
`TVM_FFI_DISABLE_TORCH_C_DLPACK=1`. Inside the container:

```bash
cp -r /opt/exl3-rocm-src /work
(cd /work && PYTORCH_ROCM_ARCH=gfx1100 python3 -m pip install \
  --no-build-isolation --force-reinstall --no-deps /work)
# install the RoPE monkey-patch from A above (exl3_rope_patch.py + .pth)
cd /tmp          # do NOT launch from /work (the source tree shadows the installed ext)
PRESET=16gb EAGER=1 MODEL=/model PORT=8000 SERVED_NAME=exl3 /usr/local/bin/serve-exl3
```

`EAGER=1` (→ `--enforce-eager`, see `presets/serve.sh`) sidesteps a separate
compiled-mode `torch._dynamo` fake-tensor `view` failure seen on this model.
Query with `POST /v1/completions` (`model=exl3`, `logprobs=5`).

### Remaining (server startup)

- The RoPE fix above is a **monkey-patch workaround**; the proper fix is a vLLM
  change to `qwen3_next.py`'s non-fused `_project_qkv_gate` path (apply the fused
  path's `positions[0]` handling). Worth a vLLM PR.
- `lm_head` (code_dim=96, k=6.0) remains an open *dequant* item from session 3
  (never exported from the 4090 due to OOM); it uses the already-correct `k=6.0`
  path, so it is expected to be fine.

## Notes / gotchas

- Run all model work inside `orcasaq2-vllm-rocm:0.30.0` (host python has no
  torch). Use `--entrypoint python`/`bash`; the image entrypoint is the server.
- The ROCm GPU is addressed with `HIP_VISIBLE_DEVICES=0` (idle GPU). The CUDA
  4090 is busy with `dreamy_bassi` — do not use it.
- `compare_dumps.py` / `inspect_dumps.py` run CPU-only (no `--device`).
- The two `.pt` dumps (`oracle_layer_states.pt`, `server_layer_states.pt`) are
  git-ignored in `dbg2/` (they are large and regenerable via the commands above);
  the regenerated files from this session were ~14 MB and ~13 MB.
- vLLM source checkout for reading the model/attention/GDN code:
  `third_party/vllm-rocm-debug` (tag v0.30.0, commit ced6857afa0ea7b2e3f0846a62e1394e90f15607).
  The installed package is `vllm 0.30.0+rocm723` in
  `/usr/local/lib/python3.12/dist-packages/vllm/...`.
- Model config (Qwen3.5 hybrid): 64 layers, `hidden_size=5120`; full attention
  every 4th layer (0-based 3,7,…,63), layers 0,1,2 are GDN. GDN:
  `linear_num_key_heads=16`, `linear_num_value_heads=48`, head_dim 128 (K_DIM==V_DIM),
  `linear_conv_kernel_dim=4`. Prompt ids `[760,6511,314,9338,369]`.
