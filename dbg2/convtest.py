import torch, torch.nn.functional as F
torch.manual_seed(0)
T, D, W = 5, 8, 4
x = torch.randn(T, D, dtype=torch.float32)
Wt = torch.randn(D, W, dtype=torch.float32)
# manual causal: out[t] = sum_j Wt[j] x[t-j]
out = torch.zeros(T, D)
for j in range(W):
    if j == 0:
        xt = x
    else:
        xt = torch.cat([torch.zeros(j, D), x[:T-j]])
    out += xt * Wt[:, j]
# conv1d reference: w[j] = Wt[3-j]
Wrev = Wt.flip(1).unsqueeze(1)
y = F.conv1d(x.float().transpose(0,1).unsqueeze(0), Wrev, groups=D)
ref = torch.zeros(T, D)
ref[W-1:] = y[0].t()
print("manual row0:", out[0].tolist())
print("ref    row0:", ref[0].tolist())
print("max diff:", (out - ref).abs().max().item())
# also check plain no-reverse
y2 = F.conv1d(x.float().transpose(0,1).unsqueeze(0), Wt.unsqueeze(1), groups=D)
ref2 = torch.zeros(T, D); ref2[W-1:] = y2[0].t()
print("max diff (no reverse):", (out - ref2).abs().max().item())
