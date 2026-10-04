"""Ground-truth check: naive per-token GDN recurrence (the decode path's math,
transcribed from fused_recurrent.py lines 120-149) in fp32 torch, compared
against (a) the current ROCm chunked-kernel output and (b) the saved CUDA
reference. Same inputs as dbg/test_fla2.py (seed 1234, T=16, K=V=128).
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

# ---- naive per-token recurrence, fp32 ----
h = h0[0].float().clone()  # [Hv,V,K]
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

rocm = torch.load("/tmp/fla2_rocm.pt", map_location="cuda")
cuda = torch.load("/work/dbg/fla2_cuda.pt", map_location="cuda")

for name, o in (("ROCm current", rocm["oA"][0].float()),
                ("CUDA saved", cuda["oA"][0].float())):
    d = (o - o_rec).abs()
    print(f"{name:12} vs recurrence: max_abs {d.max().item():.5f}  mean {d.mean().item():.6f}  frac>0.01 {(d>0.01).float().mean().item()*100:.2f}%")

d = (rocm["oA"][0].float() - cuda["oA"][0].float()).abs()
print(f"ROCm vs CUDA     : max_abs {d.max().item():.5f}  mean {d.mean().item():.6f}")
