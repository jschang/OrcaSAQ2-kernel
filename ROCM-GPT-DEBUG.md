# OrcaSAQ2 ROCm / vLLM debug handoff

This is a compact handoff for another model/agent taking a first crack at the
next ROCm vLLM diagnosis. The longer chronological notes are in
`ROCM-DEBUG.md`; this file summarizes what was just done and where the current
evidence points.

## Current symptom

Goal: serve the EXL3 OrcaSAQ2/Qwen3.5 hybrid model correctly on ROCm/gfx1100.

Known bad ROCm behavior, fresh single prompt:

```text
prompt: "The capital of France is"
max_tokens: 1
temperature: 0

CUDA/reference: top token " Paris", logprob ≈ -0.4135
ROCm target:    top token " the",   logprob ≈ -1.27
```

This still reproduces on a clean patched ROCm server. So the mrope/startup fix
below is not the final quality fix.

## What changed in the plugin

`orcasaq2/__init__.py` was patched to preserve Qwen3.5 mrope metadata that
Transformers warns about and tends to drop from the text sub-config:

- `mrope_section`
- `mrope_interleaved`

The patch wraps:

- `vllm.transformers_utils.config.get_hf_text_config`
- `vllm.config.model.get_hf_text_config`, when present
- `vllm.model_executor.models.qwen3_next.Qwen3NextAttention.__init__`

This was needed because a clean unpatched ROCm server lost those fields and
could later crash during Dynamo/fake-shape tracing with an invalid Qwen3.5
mrope/attention shape. The old debug server had equivalent logic hidden in
`dbg/sitecustomize.py`; it is now in the real plugin.

Verification inside the patched ROCm container:

```text
top rope_parameters:  has mrope_interleaved + mrope_section
text rope_parameters: has mrope_interleaved + mrope_section
```

The Transformers warning about unrecognized keys still prints. That warning is
noise if the follow-up config check shows the keys survived.

## Clean ROCm server result

A disposable clean ROCm server was launched on GPU1 as `orca-patched-rocm` with
the patched workspace installed into the image at runtime.

Important container details:

- image: `orcasaq2-vllm-rocm:0.30.0`
- base lineage: `vllm/vllm-openai-rocm:v0.30.0`
- installed vLLM reports: `0.30.0+rocm723`
- torch inside container: `2.12.0+git6bbd260`, HIP `7.2.53211`
- needed `--shm-size=16g` because KV offload wants about 8.6 GB `/dev/shm`
- served on container port 8000, host mapping requested as 8002
- API became ready after a long ROCm/vLLM cache/warmup phase

Host-side `127.0.0.1:8002` and `192.168.4.129:8002` refused from this harness,
but the API was reachable from inside the container. The in-container probe is
authoritative for the model path:

```bash
docker exec -i orca-patched-rocm bash -lc 'curl -sS -m 120 \
  http://127.0.0.1:8000/v1/completions \
  -H "Authorization: Bearer HandsomeLlm" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"exl3\",\"prompt\":\"The capital of France is\",\"max_tokens\":1,\"temperature\":0,\"logprobs\":5}"'
```

Returned:

```text
text: " the"
token_logprobs: [-1.27031409740448]
top_logprobs: " the", " a", "...", " an", " what"
```

So the original quality bug remains.

## Components exonerated or mostly exonerated

The recent pass tried to avoid another loop by checking leaf paths against
oracles rather than staring at the served output.

### Embedding

The int8 embedding gather/dequant path was compared on ROCm against a CPU bf16
oracle for gathered prompt rows.

Result:

```text
max_abs_vs_cpu_bf16 0.0
bad_gt_0 0
```

This strongly exonerates embedding sign/scale/cast math.

### Full attention backend / layer

Synthetic ROCm full-attention backend versus PyTorch SDPA:

- block table/cache slot 0 was bad because vLLM reserves block 0 as
  `NULL_BLOCK_ID`
- using cache slot 1 worked
- output matched PyTorch SDPA within bf16 noise

Actual full-attention layer diagnostic was added:

- `dbg/attn_layer3_diag.py`

It loads real layer-3 q/k/v/o EXL3 shards, q/k RMSNorm weights, applies direct
text-only RoPE, runs ROCm attention backend, compares against PyTorch SDPA, then
runs `o_proj`.

Result:

```text
attn        max_abs 0.00390625  bad_gt_1e-2 0
gated_core  max_abs 0.001953125 bad_gt_1e-2 0
o_proj_out  max_abs 0.0078125  bad_gt_1e-2 0
```

This does not prove every served attention call is perfect, but it makes the
basic full-attention math path unlikely to be the cause.

### lm_head

lm_head diagnostic was added:

- `dbg/lm_head_diag.py`

It compares fused EXL3 lm_head GEMM against reconstruct+GEMM reference over the
full vocab.

Result:

```text
fused_vs_reconstruct max_abs 0.0 mean_abs 0.0 bad_gt_1e-2 0
top tokens identical
```

lm_head EXL3 path is exonerated.

### GDN / conv / FLA context from earlier notes

From `ROCM-DEBUG.md`:

- full standalone GDN layer previously matched CUDA/ROCm within normal bf16-ish
  tolerance
- zero-state FLA path matched
- non-zero `initial_state` FLA path is a real ROCm bug, but that is expected to
  affect prefix-caching / resumed state / multi-turn rather than a fresh single
  prefill

Do not chase the old in-situ conv manual-reference mismatch blindly; the long
notes explain why it was inconclusive.

## Current interpretation

This no longer looks like a narrow `orcasaq2` EXL3 GEMM/plugin leaf-kernel bug.

The remaining likely area is the integrated vLLM ROCm execution path:

- Qwen3.5 hybrid model runner wiring
- GDN/Mamba state and metadata under real served prefill
- residual/RMSNorm sequencing/glue
- cache/block metadata interactions
- a ROCm backend behavior that appears only under the real vLLM model runner
  shapes/layout/state, not standalone oracle scripts

The right next diagnostic is to instrument the actual served prefill and dump
hidden states after selected layers, then binary-search the first divergence
against CUDA or a trusted reference.

## vLLM source checkout

Upstream vLLM was cloned for the next step:

```text
third_party/vllm-rocm-debug
```

Checkout:

```text
tag:    v0.30.0
commit: ced6857afa0ea7b2e3f0846a62e1394e90f15607
```

This matches the image tag lineage as closely as currently known:

```text
orcasaq2-vllm-rocm:0.30.0
FROM vllm/vllm-openai-rocm:v0.30.0
installed package: vllm 0.30.0+rocm723
```

The installed wheel does not expose a commit id:

```text
vllm/_version.py:
__version__ = '0.30.0'
__commit_id__ = None
```

## Suggested next step

Build an instrumentable ROCm vLLM image from `third_party/vllm-rocm-debug`,
preferably against the same ROCm/PyTorch environment as the current working
image.

Pragmatic plan:

1. Start from the existing working image or its upstream base, not from scratch.
2. Force-reinstall vLLM from the local source checkout with `VLLM_TARGET_DEVICE=rocm`.
3. Keep `PYTORCH_ROCM_ARCH=gfx1100` and a low `MAX_JOBS` if build memory becomes
   annoying.
4. Reinstall the local `orcasaq2` plugin afterward.
5. Add instrumentation directly in vLLM source around the Qwen3.5 model/layers
   to dump hidden states for the exact prompt.
6. Compare layer outputs between CUDA-good and ROCm-bad, or between standalone
   reference and served ROCm, to find the first divergence.

Useful files to inspect first in the vLLM checkout:

```text
third_party/vllm-rocm-debug/vllm/model_executor/models/qwen3_5.py
third_party/vllm-rocm-debug/vllm/model_executor/models/qwen3_next.py
third_party/vllm-rocm-debug/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
third_party/vllm-rocm-debug/vllm/v1/attention/backends/rocm_attn.py
third_party/vllm-rocm-debug/vllm/v1/attention/backends/utils.py
```

Remember: synthetic cache/block tables must not use slot 0; vLLM reserves block
0 as `NULL_BLOCK_ID`.

## Current worktree notes

Expected relevant local changes:

```text
M  orcasaq2/__init__.py
?? dbg/attn_layer3_diag.py
?? dbg/lm_head_diag.py
?? third_party/vllm-rocm-debug/
```

Generated local packaging outputs may exist:

```text
build/
orcasaq2_kernel.egg-info/
```

They are ignored by `.gitignore`.

