import torch, sys
tag = sys.argv[1] if len(sys.argv) > 1 else "out"
torch.manual_seed(11)
L = 16
H, K, HV, V = 16, 128, 48, 128
qkv_dim = 2*H*K + HV*V
conv_output = torch.randn(L, qkv_dim, device="cuda", dtype=torch.bfloat16)
a = torch.randn(L, HV, device="cuda", dtype=torch.bfloat16)
b = torch.randn(L, HV, device="cuda", dtype=torch.bfloat16)
A_log = torch.randn(HV, device="cuda", dtype=torch.float32)
dt_bias = torch.randn(HV, device="cuda", dtype=torch.float32)

# Reference split + l2norm in torch (fp32 math)
co = conv_output.float()
q_r = co[:, :H*K].reshape(L, H, K)
k_r = co[:, H*K:2*H*K].reshape(L, H, K)
v_r = co[:, 2*H*K:].reshape(L, HV, V)
q_r = q_r / (q_r.norm(dim=-1, keepdim=True) + 1e-6)
k_r = k_r / (k_r.norm(dim=-1, keepdim=True) + 1e-6)

# Load the kernel results from the other run
import os
other = "cuda" if tag=="rocm" else "rocm"
d = torch.load(f"/tmp/fla4_{other}.pt")
for name, ref, kern in [("q", q_r, d["q"]), ("k", k_r, d["k"]), ("v", v_r, d["v"])]:
    dk = (ref - kern.float()).abs().max().item()
    print(f"{tag}(ref-vs-{other}): {name} max_abs_diff={dk:.5f}")
# also save reference
torch.save({"q":q_r.cpu(),"k":k_r.cpu(),"v":v_r.cpu()}, f"/tmp/fla4ref_{tag}.pt")
