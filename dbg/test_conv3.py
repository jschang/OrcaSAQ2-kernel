import torch, sys
torch.manual_seed(3)
DIM, L, W = 8, 5, 4
x2d = torch.randn(L, DIM, device="cuda", dtype=torch.bfloat16)
x = x2d.transpose(0,1)  # [DIM, L]
print("x2d =")
print(x2d.float().detach().cpu().numpy().round(3))
weight = torch.randn(DIM, W, device="cuda", dtype=torch.bfloat16)
bias = torch.randn(DIM, device="cuda", dtype=torch.bfloat16)
conv_states = torch.zeros(2, DIM, W-1, device="cuda", dtype=torch.bfloat16)
qsl = torch.tensor([0, L], device="cuda", dtype=torch.int32)
cache_indices = torch.tensor([0], device="cuda", dtype=torch.int32)
has_init = torch.tensor([False], device="cuda")
print("weight ="); print(weight.float().detach().cpu().numpy().round(3))
print("bias ="); print(bias.float().detach().cpu().numpy().round(3))

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
out = causal_conv1d_fn(x, weight, bias, conv_states=conv_states,
    query_start_loc=qsl, cache_indices=cache_indices, has_initial_state=has_init, activation="silu")
print("out (kernel) [DIM,L] =")
print(out.float().detach().cpu().numpy().round(3))
print("out transposed [L,DIM] =")
print(out.transpose(0,1).float().detach().cpu().numpy().round(3))

# reference: out[t,d] = silu(bias[d] + sum_j weight[d,j]*x[t-j,d]) with left-zero-pad
xs = x2d.float()
wt = weight.float().t()  # [W, DIM]
ref = torch.zeros(L, DIM, device="cuda")
for t in range(L):
    acc = torch.zeros(DIM, device="cuda")
    for j in range(W):
        tt = t - (W-1-j)
        xv = xs[tt] if 0 <= tt < L else torch.zeros(DIM, device="cuda")
        acc = acc + wt[j] * xv
    ref[t] = acc
ref = ref + bias.float()
ref = torch.nn.functional.silu(ref)
print("ref (torch) [L,DIM] =")
print(ref.detach().cpu().numpy().round(3))
print("max abs diff kernel-vs-ref:", (out.transpose(0,1).float() - ref).abs().max().item())
