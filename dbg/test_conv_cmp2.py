import sys, torch
import torch.nn.functional as F

pt = torch.load(sys.argv[1], map_location="cpu")
x = pt["x"]; weight = pt["weight"]; bias = pt["bias"]; out_rocm = pt["out_rocm"]; L = pt["L"]
DIM, W = weight.shape
dev = "cuda"

x_c = x.to(dev); wt = weight.to(dev); bf = (None if bias is None else bias.to(dev)); out_rocm_c = out_rocm.to(dev)
xs = x_c.float().transpose(0, 1)  # [L, DIM]

# CORRECT manual causal conv: out[t] = sum_j w[:,j] * x[t-j]  (w[0] on newest token)
def causal_conv_ref(xtm, w, bias=None):
    L, D = xtm.shape; Wd = w.shape[1]
    out = torch.zeros(L, D, device=xtm.device, dtype=xtm.dtype)
    for j in range(Wd):
        out[j:] += w[:, j] * xtm[:L-j]
    if bias is not None:
        out += bias
    return F.silu(out)

ref = causal_conv_ref(xs, wt.float(), bf.float() if bf is not None else None).transpose(0,1).contiguous()  # [DIM, L]

# run the actual vllm conv kernel on CUDA
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
conv_state = torch.zeros(1, DIM, W-1, device=dev, dtype=x_c.dtype)
qsl = torch.tensor([0, L], device=dev, dtype=torch.int32)
ci = torch.tensor([0], device=dev, dtype=torch.int64)
his = torch.tensor([False], device=dev, dtype=torch.bool)
out_kernel = causal_conv1d_fn(x_c, wt, bf, conv_states=conv_state, query_start_loc=qsl,
                              cache_indices=ci, has_initial_state=his, activation="silu", metadata=None)

d_rocm = (ref - out_rocm_c.float()).abs()
d_kernel = (ref - out_kernel.float()).abs()
print(f"DIM={DIM} W={W} L={L}  x_absmax={x_c.float().abs().max().item():.3f}  ref_absmax={ref.float().abs().max().item():.3f}")
print(f"ROCm kernel vs CORRECT ref: max_abs_diff={d_rocm.max().item():.6f}  mean={d_rocm.mean().item():.8f}  n_gt1={int((d_rocm>1).sum().item())} / {d_rocm.numel()}")
print(f"CUDA kernel vs CORRECT ref: max_abs_diff={d_kernel.max().item():.6f}  mean={d_kernel.mean().item():.8f}  n_gt1={int((d_kernel>1).sum().item())} / {d_kernel.numel()}")
d_direct = (out_rocm_c.float() - out_kernel.float()).abs()
print(f"ROCm vs CUDA kernel direct: max_abs_diff={d_direct.max().item():.6f}  n_gt1={int((d_direct>1).sum().item())}")
print(f"ROCm per_tok_max: {[round(v,4) for v in d_rocm.amax(dim=1).tolist()]}")
print(f"CUDA per_tok_max: {[round(v,4) for v in d_kernel.amax(dim=1).tolist()]}")
