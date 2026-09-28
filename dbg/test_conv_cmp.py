import sys, torch
import torch.nn.functional as F

pt = torch.load(sys.argv[1], map_location="cpu")
x = pt["x"]          # [DIM, L] contiguous (channel-last original)
weight = pt["weight"] # [DIM, W]
bias = pt["bias"]     # [DIM]
out_rocm = pt["out_rocm"]  # [DIM, L]
L = pt["L"]
DIM, W = weight.shape
dev = "cuda"

x_c = x.to(dev)
wt = weight.to(dev)
bf = None if bias is None else bias.to(dev)
out_rocm_c = out_rocm.to(dev)

# --- torch reference (float32) ---
xs = x_c.float().transpose(0, 1)  # [L, DIM]
pad = torch.zeros(W - 1, DIM, device=dev)
inp_tm = torch.cat([pad, xs], dim=0)          # [L+W-1, DIM]
inp_cm = inp_tm.transpose(0, 1).unsqueeze(0)  # [1, DIM, L+W-1]
w3 = weight.float().to(dev).unsqueeze(1)      # [DIM, 1, W]
bf3 = None if bias is None else bias.float().to(dev)
ref = F.conv1d(inp_cm, w3, bias=bf3, groups=DIM).squeeze(0).transpose(0, 1)  # [L, DIM]
ref = F.silu(ref)
ref = ref.transpose(0, 1).contiguous()  # [DIM, L]

# --- run the actual vllm conv kernel on CUDA ---
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
conv_state = torch.zeros(1, DIM, W - 1, device=dev, dtype=x_c.dtype)
query_start_loc = torch.tensor([0, L], device=dev, dtype=torch.int32)
cache_indices = torch.tensor([0], device=dev, dtype=torch.int64)
has_initial_state = torch.tensor([False], device=dev, dtype=torch.bool)
out_kernel = causal_conv1d_fn(
    x_c, wt, bf,
    conv_states=conv_state,
    query_start_loc=query_start_loc,
    cache_indices=cache_indices,
    has_initial_state=has_initial_state,
    activation="silu",
    metadata=None,
)

d_rocm = (ref - out_rocm_c.float()).abs()
d_kernel = (ref - out_kernel.float()).abs()
print(f"DIM={DIM} W={W} L={L}")
print(f"x_absmax={x_c.float().abs().max().item():.3f}")
print(f"ref_absmax={ref.float().abs().max().item():.3f}")
print(f"ROCm kernel out vs ref: max_abs_diff={d_rocm.max().item():.5f}  mean={d_rocm.mean().item():.6f}  n_gt1={int((d_rocm>1).sum().item())}")
print(f"CUDA kernel out vs ref: max_abs_diff={d_kernel.max().item():.5f}  mean={d_kernel.mean().item():.6f}  n_gt1={int((d_kernel>1).sum().item())}")
d_direct = (out_rocm_c.float() - out_kernel.float()).abs()
print(f"ROCm vs CUDA kernel direct: max_abs_diff={d_direct.max().item():.5f}  n_gt1={int((d_direct>1).sum().item())}")
# per-token breakdown for both
print(f"ROCm per_tok_max: {[round(v,3) for v in d_rocm.amax(dim=1).tolist()]}")
print(f"CUDA per_tok_max: {[round(v,3) for v in d_kernel.amax(dim=1).tolist()]}")
