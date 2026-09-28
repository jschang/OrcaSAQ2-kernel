import torch
import torch.nn.functional as F
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

torch.manual_seed(0)
C, K, L = 8, 4, 16          # channels, kernel, tokens
num_slots = 4

x = torch.randn(C, L, device="cuda", dtype=torch.bfloat16)          # [C, L] channel-first
weight = torch.randn(C, K, device="cuda", dtype=torch.bfloat16)  # [C,K] depthwise
bias = torch.randn(C, device="cuda", dtype=torch.bfloat16)
conv_states = torch.zeros(num_slots, C, K-1, device="cuda", dtype=torch.bfloat16)
query_start_loc = torch.tensor([0, L], device="cuda", dtype=torch.int32)
has_initial_state = torch.tensor([False], device="cuda", dtype=torch.bool)
cache_indices = torch.tensor([0], device="cuda", dtype=torch.int32)

out = causal_conv1d_fn(
    x, weight, bias,
    conv_states=conv_states,
    has_initial_state=has_initial_state,
    cache_indices=cache_indices,
    query_start_loc=query_start_loc,
    activation="silu",
)
out = out.transpose(0, 1)  # -> [L, C] as model does after the call

# reference: causal depthwise conv1d (left-pad K-1) + silu
x_pad = torch.cat([torch.zeros(C, K-1, device="cuda"), x], dim=1)
ref = F.conv1d(x_pad, weight.float().unsqueeze(1), bias.float(), groups=C)[:, :L].to(torch.bfloat16)
ref = F.silu(ref)
ref = ref.transpose(0, 1)  # [L, C]

diff = (out.float() - ref.float()).abs()
print("conv1d_fn: out.shape", tuple(out.shape), "max_abs_diff", diff.max().item(),
      "mean_abs_diff", diff.mean().item())
print("out finite:", torch.isfinite(out).all().item(), "ref finite:", torch.isfinite(ref).all().item())
print("out[0][:4]   =", out[0][:4].tolist())
print("ref[0][:4]   =", ref[0][:4].tolist())
