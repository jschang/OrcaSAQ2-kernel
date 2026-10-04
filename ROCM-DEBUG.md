# OrcaSAQ2 CUDA → ROCm (gfx1100) — debug log

Goal: the model **serves** on ROCm (gfx1100) and emits **correct** output.
Current symptom (fresh prefill, `"The capital of France is"`, `max_tokens=1, temp=0`):
CUDA top `" Paris"` (logprob -0.4135); ROCm top `" the"` (-1.2283), `" Paris"` dropped
to -3.9158. So the **fresh-prefill hidden state** diverges from CUDA on ROCm.
Strategy: take one sub-component at a time, run it standalone on **both** images with
identical inputs, and diff the tensors. If it matches, that sub-component is exonerated.

---

## CURRENT STATE (2026-09-29) — read this first

**Exonerated** (byte/near-byte CUDA-vs-ROCm standalone, both running the real kernels):
- **Entire GDN layer** (in_proj→conv→post_conv→FLA→RMSNormGated→out_proj) — the 48
  GDN layers. Includes `causal_conv1d_fn` (`mixed_qkv_conv` max|diff| 0.0078, 0 bad).
- EXL3 dequant, Hadamard, `shard_gemm`, RoPE, `fused_post_conv_prep`, zero-init FLA.

**Confirmed real bug (separate, not the current symptom):**
- FLA `chunk_gated_delta_rule` **non-zero `initial_state`** path is wrong on ROCm
  (max_abs_diff 4.34; zero-state matches). Bites prefix-caching / multi-turn /
  resumed state — **not** a fresh single-seq prefill (state is zeroed there).
  **RESOLVED 10-03 (ticket #12): false alarm on the current stack.** The 4.34 diff
  was against a stale reference generated on nvidia-0's *older* vLLM image, whose
  FLA had a real h0-path bug (o stage) since fixed upstream. Current ROCm output
  matches the true gated-delta-rule math (three independent implementations agree,
  plus an E2E prefix-cache A/B on rocm-1). See the 2026-10-03 section at the
  bottom. `dbg/fla2_cuda.pt` is a **bad reference** — do not use it for compares.

**Still OPEN (never run standalone on both images) — the fresh-prefill bug is here:**
1. **int8 embedding** (`orcasaq2/heads.py`) — affects every token; cheapest test.
2. **Full-attention layer end-to-end** (qkv_proj→RoPE→**flash-attn**→o_proj) —
   only the pieces (RoPE, GEMM, output gate) verified; flash-attn core not tested.
3. **lm_head** (EXL3 trellis `apply`).
4. Layer/final RMSNorm — standard, low probability.

**Next step (ranked):** (1) diff `embedding(ids)` ROCm-vs-CUDA (cheap);
(2) full-attention layer-1 standalone compare; (3) lm_head. A **CPU oracle** is also
on the table: `fla-org/flash-linear-attention` (platform-agnostic, verified on AMD)
for the GDN math; PyTorch `scaled_dot_product_attention` on CPU for flash-attn.
Note: vLLM's own CPU backend (`vllm-cpu`) will NOT load this EXL3 checkpoint.

---

## Environment

| | CUDA reference | ROCm target |
|---|---|---|
| image | `orcasaq2-vllm:0.30.0` | `orcasaq2-vllm-rocm:0.30.0` |
| GPU | RTX 4090 (`--gpus all`, `CUDA_VISIBLE_DEVICES=0`) | gfx1100 (`--device /dev/kfd --device /dev/dri`, `HIP_VISIBLE_DEVICES=0`) |
| extra ROCm env | — | `HSA_ENABLE_IPC_MODE_LEGACY=1 HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1` |
| workdir | `-v /tmp/opencode:/data` | same |

- vLLM engine holds ~22.7 GB on the 4090 → only ~1.7 GB free, so a **full model**
  cannot be loaded on CUDA. All comparisons are **standalone single-layer** runs.
- Model: `Qwen3_5ForCausalLM`, 64 hybrid layers = **48 GDN + 16 full-attention**.
  `hidden=5120`, `vocab=248320`.
- GDN dims: `key_heads=16`, `value_heads=48`, `key_head_dim=128`, `value_head_dim=128`,
  `conv_kernel=4`.
- Checkpoint: `/mnt/sdd2/Models/orcarouter/OrcaSAQ-2-27B/` (4 safetensors shards, ~12 GB).
  Layer-0 GDN shards already extracted → `/tmp/opencode/gdn_layer0.pt` (44 MB).
- `orcasaq2` package installed at `/usr/local/lib/python3.12/dist-packages/orcasaq2`
  (no `/opt/orca` path insert needed).
- **PyTorch device string is `"cuda"` on BOTH images** (ROCm reports as `cuda`), not `"rocm"`.
- **Live containers:** `bold_taussig` (CUDA brain, port 8000, **never kill** —
  standalone single-layer runs only), `orca-rope-debug` (live ROCm server, GPU0,
  port 8001, model `exl3`), `fla-test` (ROCm kernel-test, GPU1, `HIP_VISIBLE_DEVICES=1`).
  One AMD GPU at a time. See "Live-env notes" for deploy details.

---

## What I've verified correct on ROCm (byte/near-byte vs CUDA)

| component | how | result |
|---|---|---|
| EXL3 dequant | byte-for-byte vs CUDA | ✅ exact |
| Hadamard | standalone | ✅ exact |
| GEMM / scaling | standalone | ✅ matches |
| RoPE | standalone | ✅ matches |
| `fused_post_conv_prep` (GQA) | standalone | ✅ matches |
| full-attn output gate | standalone | ✅ matches |
| `shard_gemm` op | standalone | ✅ matches |
| FLA `chunk_gated_delta_rule` (zero init) | standalone, H=16 HV=48 K=64 V=64 T=64 | ✅ rel err 0.0063 (0.63%, normal bf16) |
| FLA `chunk_gated_delta_rule` (non-zero init) | `test_fla2.py` | ❌ max_abs_diff 4.34 — real ROCm bug (zero-state OK) |
| **Full GDN layer** (in_proj→conv→post_conv→FLA→RMSNormGated→out_proj) | standalone layer-0, L=64 | ✅ see below |

### Full GDN layer (the big one — 48/64 layers)

`/tmp/opencode/gdn_standalone.py` runs layer-0 GDN end-to-end on both images and
saves intermediates; `/tmp/opencode/gdn_cmp.py` diffs them.

| tensor | max\|diff\| | rel err | bad elems |
|---|---|---|---|
| `mixed_qkv` (in_proj out) | 0.0039 | 0.07% | 0/655K |
| `mixed_qkv_conv` (conv out) | 0.0078 | 0.50% | 0/655K |
| `core` (FLA out) | 0.026 | 0.71% | 3/393K |
| `final_state` (recurrent) | 0.0016 | 0.87% | 0/786K |
| `out` (final) | 0.039 | 0.63% | 3/327K |

Everything matches within ~0.7% (normal bf16 noise; only 3 elems out of ~393K differ
by ~0.016, rounding accumulated through the chunk recurrence). **GDN is exonerated.**

---

## What threw me for a loop

1. **Standalone conv returned all-zeros (CUDA) / NaN (ROCm)** even with correct
   channel-last layout. Chased it for a long time, including an isolated test
   (`/tmp/opencode/conv_dbg2.py`).
   - **Root cause:** my synthetic `cache_indices = [0]`, but vLLM reserves block **0**
     as `NULL_BLOCK_ID` (`vllm/v1/attention/backends/utils.py:47`). The triton kernel
     `_causal_conv1d_fwd_kernel` does `if conv_states_input_coord == null_block_id: return`
     → every program bails → all-zeros. Fix: use cache **slot 1**, allocate a 2-line
     cache (`conv_state`/`ssm_state` shape `[2, ...]`), and index `ssm_state` by slot 1.
   - Lesson: the live model's block table starts at 1, so the real path never hits this.
     A synthetic harness must too.
2. **`GDNAttentionMetadata`** — easy to miss required fields (`num_prefill_tokens`,
   `num_decode_tokens`, `num_spec_decode_tokens`, …). Also needed
   `compute_causal_conv1d_metadata(...)` to populate `nums_dict`/`batch_ptr`/
   `token_chunk_offset_ptr` for the conv kernel.
3. **Import paths** drifted between vLLM versions — `prepare_chunk_indices/offsets`
   live in `vllm.third_party.flash_linear_attention.ops.index`, `FLA_CHUNK_SIZE` in
   `...ops.utils`, `fused_post_conv_prep` in
   `vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn`, not in `mamba.ops`.
4. `mcg`/`mul1` flags: checkpoint `in_proj_qkv`/`in_proj_z` carry `mul1` but **no**
   `mcg` → `mcg=False, mul1=True`. `conv1d` has **no bias** → use zero bias.

---

## Top remaining suspect (NOT yet tested): int8 embedding

`orcasaq2/heads.py` → `Exl3EmbeddingMethod` dequants the embedding **on the gathered
rows** (table stays int8):

```python
rows = layer.exl3_embed_q[input_].to(layer.exl3_embed_s.dtype)
return rows * layer.exl3_embed_s[input_]
```

- `exl3_embed_s = s.to(dev, params_dtype).reshape(-1, 1)` (per-row scale, model dtype).
- The ablation comment claims int8 embed "scored KLD 0.0338 vs 0.0341 for bf16"
  (i.e. treated as free) — but that was the **CUDA/exllamav3** side.
- **Why it fits the symptom:** if the per-row scale (dtype, shape, or the int8 sign
  handling) is subtly off, *every* token embeds slightly wrong → coherent but
  consistently wrong text, exactly what we see. This is a much stronger match than
  the 16/64 full-attn layers.
- **Quick test:** compare `embedding(input_ids)` on both images for a batch of ids.
  Cheap, no KV cache needed. Also verify the *head* (`apply` path, the matmul) since
  it shares the int8 table.
- The exllamav3 twin patch `orcasaq2/patches/int8_embedding.py` has a documented
  silent-failure mode (deferred load → table reads as all-zeros → "a stream of !").
  That's a different failure (garbage), not "coherent-but-wrong", but worth confirming
  the vLLM path's `process_weights_after_loading` actually loaded `qweight`+`scales`
  (not the dense `weight`) — a wrong branch would silently change the table.

---

## Next step (chosen): full-attention layer standalone compare

Take one full-attention layer (e.g. layer-1, the first full-attn after GDN layer-0)
and compare on both images: qkv_proj (EXL3 `shard_gemm`) → q/k/v split+reshape →
RoPE → flash-attention (with a 1-seq KV cache) → o_proj (EXL3). Save intermediates,
diff like the GDN test. Need to extract layer-1 full-attn EXL3 shards from the
safetensors checkpoint.

Fallback / parallel: the int8 embedding test above is cheaper and is the more likely
culprit — run it first if quick.

---

## Files

All debug scripts + saved tensors are now in the repo at `dbg/` (pulled from
`/tmp/opencode/` on 2026-09-29). Host `/tmp/opencode` copies still exist too.

GDN standalone (the proven-correct harness / template for new tests) — **LOST**:
- `gdn_standalone.py`, `gdn_cmp.py`, `gdn_layer0.pt`, `gdn_cuda.pt`, `gdn_rocm.pt`
  were at `/tmp/opencode/` but are now gone (not in host `/tmp/opencode`, any
  container `/tmp`, or `/data`). The GDN *result* is still documented above
  (byte-verified); only the harness is lost. Closest templates in `dbg/`:
  `test_fla*.py` (FLA kernel call) + `test_conv1d*.py` (conv kernel call).

Conv (all INCONCLUSIVE — see "post-compaction" section):
- `dbg/test_conv1d.py`, `test_conv1d2.py`, `test_conv2.py`, `test_conv3.py` —
  isolated conv tests. (`conv_dbg.py`/`conv_dbg2.py`, the NULL_BLOCK_ID repro,
  are also lost.)
- `dbg/test_conv_cmp.py`, `test_conv_cmp2.py`, `test_conv_cmp3.py` — live conv cmp.
- `dbg/conv_layer0_rocm.pt` — saved live conv tensors (x lost stride `(1,16384)`).
- `dbg/sitecustomize.py` — live debug harness (bind-mounted into `orca-rope-debug`;
  `docker restart` to reload).

FLA / chunk (ticket #12 — RESOLVED 10-03, see bottom; `fla2_cuda.pt` is a BAD
reference from the old nvidia-0 image):
- `dbg/test_fla.py`, `test_fla_chunk.py` + `fla_rocm.pt`, `fla_cuda.pt` — chunk test.
- `dbg/test_fla2.py` + `fla2_rocm.pt`, `fla2_cuda.pt` — non-zero-init isolation.
- `dbg2/fla_chunk_bisect.py` — stage-level chunk bisect vs independent torch
  reference (args: `T tag g_scale w_scale h0_scale h0_dtype`).
- `dbg2/fla_recurrence_check.py` — naive per-token fp32 recurrence (ground truth)
  vs ROCm output and vs the saved CUDA reference.
- `dbg2/fla_three_way.py` — chunk vs decode (Triton) vs recurrence, same inputs.
- `dbg2/gdn_prefix_ab.py` — E2E prefix-cache A/B (cold / hit / cold-after-restart)
  against the live server (args: `phase1|phase2 [base_url]`).
- `dbg/test_fla3.py` — decode (invalid, zeros both backends).
- `dbg/test_fla4.py` + `test_fla4ref.py` — `fused_post_conv_prep` (correct).
- `dbg/test_exl3gemm.py`, `test_l2norm.py`, `probe_*.py` — misc probes.

Source / config:
- `orcasaq2/heads.py` — vLLM int8 embedding + lm_head (top suspect).
- `orcasaq2/patches/int8_embedding.py` — exllamav3 int8 embedding patch (twin).
- `orcasaq2/config.py`, `orcasaq2/ops.py` — EXL3 config / `shard_gemm` op.
- `docker/Dockerfile-rocm`, `docker/build-rocm.sh` — ROCm image build.
- `~/orca-forensics/logs/debug-rope.log` — live ROCm log.

---
---

## 2026-09-29 — post-compaction: re-baselining (BREAK THE LOOP)

Live symptom (unchanged): fresh short prompt `"The capital of France is"`,
`max_tokens=1, temperature=0` → CUDA top `" Paris"` (logprob -0.4135);
ROCm top `" the"` (-1.2283), `" Paris"` dropped to -3.9158. So the **fresh
prefill** hidden state is wrong on ROCm.

### A. The live in-situ conv diagnosis is INCONCLUSIVE — do NOT chase it
I added a live conv hook (`/tmp/opencode/sitecustomize.py`, bind-mounted into
`orca-rope-debug` at `/usr/lib/python3.12/sitecustomize.py`; reload requires
`docker restart orca-rope-debug`). It captured the real layer-0 prefill conv
call (`DIM=10240 W=4 L=8`, `x_stride=(1, 16384)`, `bias=None`) and compared the
ROCm `causal_conv1d_fn` output against a **manual** torch causal-conv
reference. Result: `max_abs_diff=25.6, mean=0.0826, n_gt1=680/81920` — mostly
small per-feature diffs with a few outliers.

Why it is NOT proof the conv is broken:
- I cannot validate the manual reference. A standalone `causal_conv1d_fn` run
  on `bold_taussig` (CUDA) with the saved inputs returns `inf/nan/1e37` **even
  after reconstructing the exact channel-last stride `(1, 16384)`**. The kernel
  needs real `GDNAttentionMetadata` (`block_idx_first/last_scheduled_token`,
  `num_computed_tokens`, `nums_dict/batch_ptr/token_chunk_offset_ptr` from
  `compute_causal_conv1d_metadata`); `metadata=None` is insufficient. So there
  is no clean standalone CUDA reference to diff against.
- The RELIABLE comparison already exists: the old full-GDN standalone
  (`gdn_standalone.py`, CUDA vs ROCm, both running the real conv kernel) showed
  `mixed_qkv_conv` max|diff| **0.0078, 0 bad elems**. **Conv is exonerated.**
- The manual reference ordering was fixed this session (real kernel is a
  standard causal conv: weight tap `j` multiplies `x[t-j]`, i.e. `w[0]` on the
  *newest* token; `F.conv1d` does the opposite, so it needs the weight reversed
  or a manual loop). Even so, the 25.6 outlier cannot be attributed without a
  validated reference.

**Decision: stop debugging the conv. Treat GDN prefill (conv + post_conv) as
correct per the byte-level standalone test.**

### B. Confirmed real bug: FLA `chunk_gated_delta_rule` non-zero initial_state
Isolated with `test_fla2.py` (H=16 HV=48 K=64 V=64), ROCm vs CUDA:
- `oA` (non-zero `initial_state` path) max_abs_diff **4.34375**, mean 0.1015 — WRONG.
- `oB`/`oC` (zero-state) max_abs_diff 0.0039 — match (bf16 noise).
So: **zero-state chunk is correct; a non-zero `initial_state` is mishandled on
ROCm.** For a truly fresh single-seq prefill `initial_state` is zeroed
(`qwen_gdn_linear_attn.py` ~1526-1527), so this bug likely does NOT explain the
fresh-prefill symptom — it *would* bite prefix-caching / multi-turn / resumed
state. Still a real bug to fix, but probably not the current one.

### C. Where the bug actually is (re-focus)
Given the full GDN layer was byte-verified and conv/post-conv are exonerated,
the fresh-prefill divergence must be in a component NOT covered by that
standalone test. Ranked suspects:
1. **Full-attention layers (16/64)** — EXL3 `shard_gemm` qkv/o_proj + flash-attn
   on ROCm. The old log's "next step" was exactly this; it was never completed.
2. **int8 embedding** (`orcasaq2/heads.py`) — old log's top suspect; a per-row
   scale / sign / deferred-load issue would make *every* token embed slightly
   wrong → coherent-but-wrong. Cheap to test (compare `embedding(ids)` both
   images).
3. **lm_head** (int8) — same table as embedding; verify the `apply` matmul path.

### D. Immediate next step
Run the **full-attention layer-1 standalone CUDA-vs-ROCm** compare (extract
layer-1 EXL3 shards from the checkpoint, run qkv_proj → RoPE → flash-attn with
a 1-seq KV cache → o_proj, diff intermediates like `gdn_cmp.py`). In parallel,
the **int8 embedding** compare is cheaper — do it first if quick.

### Live-env notes (this session)
- `bold_taussig` (CUDA brain, port 8000) must **never** be killed; use it only
  for standalone single-layer kernel runs (full model doesn't fit: ~1.7 GB free).
- `orca-rope-debug` = live ROCm server, GPU0, port 8001, model `exl3`,
  `--shm-size=16g` required for `--kv-offloading-size 8`.
- `fla-test` = ROCm kernel-test container, GPU1, `HIP_VISIBLE_DEVICES=1`.
- Deploy debug code by editing host `/tmp/opencode/sitecustomize.py` (bind
  mount) then `docker restart orca-rope-debug` (direct `docker cp` to the
  container path fails: `unlinkat ... device or resource busy`).
- Debug log: `~/orca-forensics/logs/debug-rope.log`.
- Scratch: `test_conv_cmp.py/2/3.py` (conv cmp, all inconclusive),
  `test_fla2.py` + `fla2_rocm.pt`/`fla2_cuda.pt` (chunk bug), `test_fla3.py`
  (decode, invalid/zeros), `test_fla4.py` + `/tmp/fla4ref.py` (post_conv).

---
---

## 2026-10-03 — ticket #12 RESOLVED: the h0 "bug" is a false alarm on the current stack

**Bottom line: the current ROCm (vLLM 0.30.x) FLA `chunk_gated_delta_rule`
non-zero-`initial_state` path is correct.** The 4.34 mismatch found on 09-29 was
against a stale/bad CUDA reference (`dbg/fla2_cuda.pt`), not against the true
math. The saved reference was generated on nvidia-0's *older* vLLM image, whose
vendored FLA had a real h0-path bug (in the o stage) that has since been fixed
upstream.

### Evidence

1. **Unit** (dev box `orcasaq2-gemv-dev`, card 0, seed 1234, Hq=16 Hv=48 K=V=128):
   - Current ROCm chunk kernel (non-zero h0) vs an independent naive per-token
     fp32 torch recurrence (transcribed from the production decode kernel):
     max_abs **0.0082** — pure bf16 rounding. Holds at T=16/48/64/80/256
     (partial chunks, full chunks, 4-chunk state chain) with both fp32 and bf16
     h0 (`dbg2/fla_chunk_bisect.py`, `dbg2/fla_recurrence_check.py`).
   - Three-way: chunk (Triton) vs fused-recurrent decode (Triton, the production
     decode path, non-zero h0) vs naive torch recurrence — all agree within bf16
     (chunk-vs-decode max 0.0156; each vs recurrence ≤ 0.0082).
     (`dbg2/fla_three_way.py`)
   - Every intermediate stage (h chain, v_new, final_state, o) matches an
     independent stage-by-stage reference; `h[0] == h0` bitwise in every run.
   - **Saved CUDA reference vs the same naive recurrence: max_abs 4.3488, 45% of
     elements off.** The "reference" is the one that is wrong.
   - Cross-check across the saved session-1 pairs: `fla_cuda.pt`/`fla_rocm.pt`
     (h0 non-zero, model shapes) — final states agree across images (0.0063,
     bf16 noise) while o diverges by 4.34; zero-state o (`fla2` oB/oC) agrees
     (0.0039). The cross-image divergence is specific to the h0 path of the o
     stage — exactly where the old nvidia-0 FLA differs from the 0.30.0 FLA.

2. **Production E2E** (`orcasaq2-rocm-1`, prefix caching on; cache blocks are
   1568 tokens due to mamba-page alignment): `dbg2/gdn_prefix_ab.py`
   - 3818-token prompt. Cold run A (zero-state prefill). Warm run B:
     `vllm:prefix_cache_hits_total` jumped by **3136** (2 full blocks) → the tail
     682 tokens were prefilled with **non-zero initial_state** (multi-chunk h0
     path).
   - A and B outputs (reasoning + answer, greedy, 270 tokens) are
     **byte-identical**; after a container restart, cold run C is also
     byte-identical. All four factual questions about the cached prefix were
     answered correctly.

### What this means
- Nothing to fix on the ROCm side; no FLA patch needs baking into the image.
- The h0 path is verified for T=16..256, fp32/bf16 state, partial + full +
  multi-chunk. The `dbg2/` harnesses are committed as the regression suite.
- nvidia-0's older image may still carry the old h0 bug — its multi-turn /
  prefix-hit output could be wrong. Not verifiable from here (no-touch
  container). If the CUDA card's multi-turn matters, re-run
  `dbg2/fla_recurrence_check.py` and `dbg2/gdn_prefix_ab.py` there and
  regenerate the references.
- `dbg/fla2_cuda.pt` (and the o parts of `fla_cuda.pt`) are **bad references** —
  kept as evidence, do not use them for comparisons.
