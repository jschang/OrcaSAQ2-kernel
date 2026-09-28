import torch

torch.manual_seed(0)
H, T = 2, 5
K = V = 4   # real model has K_DIM==V_DIM==128, so buggy code runs but is wrong
dev = "cpu"

q = torch.randn(T, H, K)
k = torch.randn(T, H, K)
v = torch.randn(T, H, V)
g = -torch.rand(T, H)          # log-decay (negative)
beta = torch.rand(T, H)

# ---- oracle's CURRENT (buggy) recurrence ----
def buggy(q, k, v, g, beta):
    S = torch.zeros(H, K, V)
    o = torch.empty(T, H, V)
    for t in range(T):
        S = S * torch.exp(g[t]).view(H, 1, 1)
        v_new = v[t] - torch.bmm(k[t].unsqueeze(1), S).squeeze(1)
        v_new = v_new * beta[t].view(H, 1)
        S = S + v_new.unsqueeze(2) * k[t].unsqueeze(-1)      # buggy update
        o[t] = torch.einsum("hkv,hv->hk", S, q[t])             # buggy output
    return o, S

# ---- CORRECT standard GDN (fla) recurrence ----
def correct(q, k, v, g, beta):
    S = torch.zeros(H, K, V)
    o = torch.empty(T, H, V)
    for t in range(T):
        S = S * torch.exp(g[t]).view(H, 1, 1)
        v_new = v[t] - torch.einsum("hk,hkv->hv", k[t], S)    # pred = k @ S
        v_new = v_new * beta[t].view(H, 1)
        S = S + torch.einsum("hk,hv->hkv", k[t], v_new)       # S += outer(k, v_new)
        o[t] = torch.einsum("hkv,hk->hv", S, q[t])             # o = S @ q
    return o, S

ob, Sb = buggy(q, k, v, g, beta)
oc, Sc = correct(q, k, v, g, beta)

print("output  buggy-vs-correct max|diff|:", (ob - oc).abs().max().item())
print("state   buggy-vs-correct max|diff|:", (Sb - Sc).abs().max().item())
print("buggy o[0] head0:", ob[0, 0].tolist())
print("correct o[0] head0:", oc[0, 0].tolist())

# Show that the buggy update is element-wise (rank collapse) not an outer product:
t = 0
S = torch.zeros(H, K, V); S = S * torch.exp(g[0]).view(H,1,1)
v_new = v[0] - torch.bmm(k[0].unsqueeze(1), S).squeeze(1)
v_new = v_new * beta[0].view(H, 1)
upd_buggy = v_new.unsqueeze(2) * k[0].unsqueeze(-1)
print("buggy update shape:", tuple(upd_buggy.shape), " (expected outer (H,K,V)=", (H,K,V), ")")
print("buggy update is element-wise along shared axis -> broadcast adds same row to every v")
