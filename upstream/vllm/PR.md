# vLLM — Fix Qwen3.5 / Qwen3-Next RoPE 2-D `positions` crash on non-CUDA platforms

**Upstream:** `vllm-project/vllm`
**Target version:** v0.30.0 (patch generated and `git apply --check`-verified against the v0.30.0 tree)
**File:** `vllm/model_executor/models/qwen3_next.py`, function `Qwen3NextAttention._project_qkv_gate`
**Patch:** [`0001-vllm-qwen3-next-rope-2d-positions.patch`](0001-vllm-qwen3-next-rope-2d-positions.patch) (single hunk, applies with `git apply` / `patch -p1`)

## Symptom

On **ROCm** (and any non-CUDA platform), serving Qwen3.5 / Qwen3-Next aborts
during the engine's memory-profile dummy run:

```text
RuntimeError: ... rotary_embedding, pos_encoding_kernels.hip:141,
    query, key and positions must have the same batch_size and seq_len
```

The engine core fails to initialize, so the model can never be served.

## Root cause

`_project_qkv_gate` dispatches between a fused Triton path and an eager
(non-fused) split + QK-RMSNorm + RoPE path. The two branches handle 2-D
`positions` differently:

- **Fused branch** — gated on `current_platform.is_cuda()` — reduces 2-D
  positions for the non-mrope case before RoPE:
  ```python
  if positions.ndim == 2 and not getattr(self.rotary_emb, "mrope_section", None):
      positions = positions[0]
  ```
  This is correct because the profile/dummy run hands the layer `positions` of
  shape `[num_reqs, seq_len]` while `q`/`k` are already flattened to
  `[num_tokens, hidden]`.

- **Non-fused branch** — the one taken on ROCm, since the fused kernel requires
  `is_cuda()` — **never** applies that reduction. It passes the 2-D `positions`
  straight to `self.rotary_emb(positions, q, k)`, whose 1-D RoPE kernel sees a
  token-count mismatch and raises.

Observed shapes in the profile run: `positions=(3, 2048)`, `query=(2048, 6144)`,
`key=(2048, 1024)` — i.e. `positions` is 2-D while `q`/`k` are 2-D flattened to
`num_tokens=2048` rows.

## Fix

Apply the identical reduction in the non-fused branch, mirroring the fused
branch:

```python
if positions.ndim == 2 and not getattr(
    self.rotary_emb, "mrope_section", None
):
    positions = positions[0]
q, k = self.rotary_emb(positions, q, k)
```

- The `not mrope_section` guard exactly matches the fused branch, so the
  multimodal (mrope) case — where 2-D positions are meaningful — is left
  untouched.
- In normal 1-D serving, `positions.ndim == 1` and nothing changes.

## Verification

- The reduction is the same one the deployed in-container workaround applied, and
  with it the Qwen3.5 profile/dummy run proceeds, the engine boots, and the
  model serves. Verified on ROCm gfx1100, vLLM v0.30.0, OrcaSAQ-2-27B: end-to-end
  generation returns the correct top token for `"The capital of France is"`
  (`" Paris"`).
- `git apply --check` passes against the v0.30.0 tree; the fixed file
  byte-compiles.

## Scope / notes

- Minimal, localized: one guard in one branch, made consistent with the existing
  fused branch. No changes to the RoPE kernel or to the mrope path.
- CUDA is unaffected (it uses the fused branch, which already handles 2-D
  positions). The fix only activates on the non-fused path.
