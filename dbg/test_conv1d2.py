import torch
import torch.nn.functional as F
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

def ref_conv(x_cf, weight, bias, K):
    C, L = x_cf.shape
    x_pad = torch.cat([torch.zeros(C, K-1, device="cuda"), x_cf], dim=1)
    r = F.conv1d(x_pad, weight.float().unsqueeze(1), bias.float(), groups=C)[:, :L]
    return F.silu(r)

torch.manual_seed(0)
C, K, L = 8, 4, 16
num_slots = 4
weight = torch.randn(C, K, device="cuda", dtype=torch.bfloat16)
bias = torch.randn(C, device="cuda", dtype=torch.bfloat16)
qsl = torch.tensor([0, L], device="cuda", dtype=torch.int32)
has_init = torch.tensor([False], device="cuda", dtype=torch.bool)
cache_idx = torch.tensor([0], device="cuda", dtype=torch.int32)

for name, x in [
    ("channel-first [C,L] contig", torch.randn(C, L, device="cuda", dtype=torch.bfloat16)),
    ("channel-last  (model layout)", torch.randn(L, C, device="cuda", dtype=torch.bfloat16).transpose(0,1)),
]:
    conv_states = torch.zeros(num_slots, C, K-1, device="cuda", dtype=torch.bfloat16)
    out = causal_conv1d_fn(x, weight, bias, conv_states=conv_states,
                           has_initial_state=has_init, cache_indices=cache_idx,
                           query_start_loc=qsl, activation="silu")
    out_cf = out if out.shape[0]==C else out.transpose(0,1)   # to [C,L]
    ref = ref_conv(x.contiguous(), weight, bias, K)
    d = (out_cf.float()-ref.float()).abs()
    print(f"--- {name} ---")
    print("  max_abs_diff", d.max().item(), "mean_abs_diff", d.mean().item())
    print("  out row0 [", out_cf[0].tolist(), "]")
    print("  ref row0 [", ref[0].tolist(), "]")
    print("  out row1 [", out_cf[1].tolist(), "]")
    print("  ref row1 [", ref[1].tolist(), "]")
