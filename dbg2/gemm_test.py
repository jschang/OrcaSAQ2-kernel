import json, os, torch
import torch.nn.functional as F
from safetensors import safe_open
from exllamav3.ext import exllamav3_ext as ext
from orcasaq2 import reconstruct_weight

MODEL = os.environ.get("MODEL_PATH", "/model")
DEV = torch.device("cuda", 0)
BF16 = torch.bfloat16
IDX = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]

def lt(key, dtype=None):
    with safe_open(os.path.join(MODEL, IDX[key]), framework="pt", device="cpu") as f:
        t = f.get_tensor(key)
    if dtype is not None:
        t = t.to(dtype)
    return t.to(DEV).contiguous()

def gemm_compare(base, x):
    trellis = lt(base + ".trellis")
    suh = lt(base + ".suh", torch.float16)
    svh = lt(base + ".svh", torch.float16)
    has_mcg = base + ".mcg" in IDX
    has_mul1 = base + ".mul1" in IDX
    mcg = lt(base + ".mcg") if has_mcg else None
    mul1 = lt(base + ".mul1") if has_mul1 else None
    oc = svh.numel()
    W = reconstruct_weight(trellis, suh, svh, mcg, mul1).to(DEV)  # (oc, in) fp16
    rows = x.numel() // x.shape[-1]
    xf = x.reshape(rows, x.shape[-1]).half().contiguous()
    y = torch.empty((rows, oc), dtype=torch.float16, device=DEV)
    ext.exl3_gemm(xf, trellis, y, suh, torch.empty_like(xf), svh, -1,
                  mcg, mul1, 0)
    y_exl3 = y.float()
    y_dense = xf.float() @ W.float().t()
    d = (y_exl3 - y_dense).abs()
    cos = F.cosine_similarity(y_exl3.flatten(), y_dense.flatten(), dim=0).item()
    rel = (d / y_dense.abs().clamp_min(1e-3)).max().item()
    print(f"{base.split('layers.')[-1]:>34}  max={d.max().item():.4f} mean={d.mean().item():.5f} "
          f"cos={cos:.6f} relmax={rel:.4f} |y|={y_dense.abs().max().item():.3f}")

torch.manual_seed(0)
b0 = "model.language_model.layers.0"
b3 = "model.language_model.layers.3"
x5 = torch.randn(5, 5120, dtype=BF16, device=DEV)
print("=== EXL3 GEMM vs dense (identical bf16 input) ===")
gemm_compare(b0 + ".linear_attn.in_proj_qkv", x5)
gemm_compare(b0 + ".linear_attn.in_proj_z", x5)
gemm_compare(b0 + ".linear_attn.out_proj", torch.randn(5, 6144, dtype=BF16, device=DEV))
gemm_compare(b3 + ".self_attn.q_proj", x5)
gemm_compare(b3 + ".self_attn.k_proj", x5)
gemm_compare(b0 + ".mlp.gate_proj", x5)
gemm_compare(b0 + ".mlp.down_proj", torch.randn(5, 17408, dtype=BF16, device=DEV))
print("=== EXL3 GEMM vs dense M=1/2/4/8 (fused GEMV path) ===")
for rows in (1, 2, 4, 8):
    xr = torch.randn(rows, 5120, dtype=BF16, device=DEV)
    gemm_compare(b0 + ".linear_attn.in_proj_qkv", xr)
    gemm_compare(b0 + ".mlp.gate_proj", xr)
    gemm_compare(b3 + ".self_attn.q_proj", xr)
    gemm_compare(b0 + ".mlp.down_proj", torch.randn(rows, 17408, dtype=BF16, device=DEV))
print("done")
