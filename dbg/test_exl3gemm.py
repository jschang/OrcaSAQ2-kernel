import json, glob
import torch
from safetensors import safe_open
from exllamav3.ext import exllamav3_ext as ext

dev = torch.device("cuda:0")
P = "/model/"
idx = json.load(open(P + "model.safetensors.index.json"))["weight_map"]
base = "model.language_model.layers.0.linear_attn.in_proj_qkv."
def load(key):
    fn = P + idx[key]
    with safe_open(fn, framework="pt") as f:
        return f.get_tensor(key).to(dev)

trellis = load(base+"trellis")
suh = load(base+"suh")
svh = load(base+"svh")
mul1 = load(base+"mul1")
mcg = None
print("trellis", trellis.shape, trellis.dtype, "suh", suh.shape, "svh", svh.shape, "mul1", mul1.shape)
ic, oc = suh.numel(), svh.numel()
k = trellis.shape[-1]/16
k = int(k) if float(k).is_integer() else k
print("ic,oc,k =", ic, oc, k)

rows = 16
x = (torch.randn(rows, ic, device=dev, dtype=torch.half)*0.3).contiguous()

# fused path
y = torch.empty((rows, oc), dtype=torch.half, device=dev)
ext.exl3_gemm(x, trellis, y, suh, torch.empty_like(x), svh, -1, mcg is not None, mul1 is not None, 0)
yf = y.clone()

# reference path (reconstruct + hgemm_recon)
xh = torch.empty_like(x)
ext.had_r_128(x, xh, suh, None, 1.0)
w = torch.empty((ic, oc), dtype=torch.half, device=dev)
ext.reconstruct(w, trellis, k, mcg is not None, mul1 is not None)
yr = torch.empty((rows, oc), dtype=torch.half, device=dev)
ext.hgemm_recon(xh, w, yr)
ext.had_r_128(yr, yr, None, svh, 1.0)

d = (yf.float()-yr.float())
print("max_abs_diff =", d.abs().max().item())
print("rel (max abs diff / max |ref|) =", d.abs().max().item()/yr.float().abs().max().item())
print("yf sample:", yf[0,:6].tolist())
print("yr sample:", yr[0,:6].tolist())
