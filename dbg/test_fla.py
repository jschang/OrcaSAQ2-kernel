import torch, os, sys
print("device:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
torch.manual_seed(0)

from vllm.third_party.flash_linear_attention.ops import (
    chunk_gated_delta_rule as fla_chunk,
    fused_recurrent_gated_delta_rule_packed_decode as fla_recurrent,
)

# small shapes consistent with a GDN layer
B, H, K, V, L = 1, 2, 64, 64, 32
q = torch.randn(B, L, H, K, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, L, H, K, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, L, H, V, device="cuda", dtype=torch.bfloat16)
g = torch.randn(B, L, H, device="cuda", dtype=torch.float32) * 0.1
beta = torch.rand(B, L, H, device="cuda", dtype=torch.float32)
initial_state = torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32)

try:
    out, final_state = fla_chunk(q, k, v, g, beta, initial_state=initial_state)
    print("chunk OK  out.shape", tuple(out.shape), "final.shape", tuple(final_state.shape))
    print("chunk out finite:", torch.isfinite(out).all().item(),
          "final finite:", torch.isfinite(final_state).all().item())
    print("chunk out mean/absmax:", out.float().mean().item(), out.float().abs().max().item())
except Exception as e:
    print("chunk FAILED:", repr(e))

try:
    # decode path: single new token appended
    qd = q[:, -1:, :, :]
    kd = k[:, -1:, :, :]
    vd = v[:, -1:, :, :]
    gd = g[:, -1:, :]
    bd = beta[:, -1:, :]
    o2, fs2 = fla_recurrent(qd, kd, vd, gd, bd, initial_state=initial_state)
    print("recurrent OK out.shape", tuple(o2.shape), "final.shape", tuple(fs2.shape))
    print("recurrent out finite:", torch.isfinite(o2).all().item(),
          "final finite:", torch.isfinite(fs2).all().item())
    print("recurrent out mean/absmax:", o2.float().mean().item(), o2.float().abs().max().item())
except Exception as e:
    print("recurrent FAILED:", repr(e))
