"""Three-way cross-check, same inputs as dbg/test_fla2.py (seed 1234, T=16):
  1. chunked Triton kernel (the prefill path, h0 non-zero)
  2. fused-recurrent Triton kernel (the production decode path, h0 non-zero)
  3. naive per-token torch recurrence (independent math, fp32)
All three must agree within bf16 noise for the h0 path to be trusted.
"""
import torch

torch.manual_seed(1234)
B, L = 1, 16
Hq, Hv, K, V = 16, 48, 128, 128
q = torch.randn(B, L, Hq, K, device="cuda", dtype=torch.bfloat16)
k = torch.nn.functional.normalize(
    torch.randn(B, L, Hq, K, device="cuda", dtype=torch.bfloat16), p=2, dim=-1
)
v = torch.randn(B, L, Hv, V, device="cuda", dtype=torch.bfloat16)
g = torch.nn.functional.logsigmoid(torch.rand(B, L, Hv, device="cuda", dtype=torch.float32))
beta = torch.rand(B, L, Hv, device="cuda", dtype=torch.bfloat16).sigmoid()
h0 = torch.randn(B, Hv, V, K, device="cuda", dtype=torch.float32)
scale = K ** -0.5
cu = torch.tensor([0, L], device="cuda", dtype=torch.int32)

from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule as fla_chunk
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_fwd,
)

o_chunk, _ = fla_chunk(
    q, k, v, g, beta, initial_state=h0, output_final_state=True, cu_seqlens=cu
)
o_dec, _ = fused_recurrent_gated_delta_rule_fwd(
    q, k, v, g, beta,
    scale=scale,
    initial_state=h0.clone(),
    inplace_final_state=False,
    cu_seqlens=cu,
)

# naive per-token recurrence (decode-kernel math), fp32
h = h0[0].float().clone()
o_rec = torch.empty(L, Hv, V, dtype=torch.float32, device="cuda")
for t in range(L):
    for i_h in range(Hv):
        i_k = i_h // (Hv // Hq)
        hi = h[i_h] * torch.exp(g[0, t, i_h])
        kn = k[0, t, i_k].float()
        vnew = (v[0, t, i_h].float() - kn @ hi.T) * beta[0, t, i_h].float()
        hi = hi + vnew.unsqueeze(1) * kn.unsqueeze(0)
        h[i_h] = hi
        o_rec[t, i_h] = q[0, t, i_k].float() @ hi.T * scale

for name, o in (
    ("chunk (Triton)", o_chunk[0].float()),
    ("decode (Triton)", o_dec[0].float()),
    ("recurrence (torch)", o_rec),
):
    d_cd = (o - o_chunk[0].float()).abs()
    d_dd = (o - o_dec[0].float()).abs()
    d_rr = (o - o_rec).abs()
    print(f"{name:18} vs chunk: {d_cd.max().item():.5f}  "
          f"vs decode: {d_dd.max().item():.5f}  vs recurrence: {d_rr.max().item():.5f}")
