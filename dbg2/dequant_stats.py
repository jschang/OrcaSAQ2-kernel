import json, os, torch
from safetensors import safe_open
from orcasaq2 import reconstruct_weight
MODEL = "/model"; DEV = torch.device("cuda",0)
IDX = json.load(open(os.path.join(MODEL,"model.safetensors.index.json")))["weight_map"]
def lt(key,dtype=None):
    with safe_open(os.path.join(MODEL,IDX[key]),framework="pt",device="cpu") as f: t=f.get_tensor(key)
    if dtype is not None: t=t.to(dtype)
    return t.to(DEV).contiguous()
def stats(base):
    tr=lt(base+".trellis"); suh=lt(base+".suh",torch.float16); svh=lt(base+".svh",torch.float16)
    mcg=lt(base+".mcg") if base+".mcg" in IDX else None
    mul1=lt(base+".mul1") if base+".mul1" in IDX else None
    W=reconstruct_weight(tr,suh,svh,mcg,mul1).float()
    print(f"{base.split('layers.')[-1]:>32}: shape={tuple(W.shape)} mean={W.mean().item():.5f} std={W.std().item():.5f} "
          f"max={W.abs().max().item():.4f} min={W.min().item():.4f} frac_zero={(W==0).float().mean().item():.3f}")
for b in ["model.language_model.layers.0.linear_attn.in_proj_qkv",
          "model.language_model.layers.0.mlp.gate_proj",
          "model.language_model.layers.3.self_attn.q_proj",
          "model.language_model.embed_tokens"]:
    stats(b)
