"""Run ONLY ext.reconstruct (no Hadamard, no suh/svh scaling) on one projection and
save the raw dequantized (ic, oc) half tensor. Device-agnostic: runs on CUDA
(dreamy_bassi) or ROCm (the rocm image) and writes an identical file, so a diff of the
two isolates the reconstruct kernel itself.

  MODEL_PATH=/model OUT=/tmp/w_raw.pt [PREFIX=model.language_model.layers.3.self_attn.v_proj] \
      python reconstruct_raw.py
"""
import os
import json

import torch
from safetensors import safe_open
from exllamav3.ext import exllamav3_ext as ext

MODEL = os.environ.get("MODEL_PATH", "/model")
OUT = os.environ.get("OUT", "w_raw.pt")
PREFIX = os.environ.get("PREFIX", "model.language_model.layers.3.self_attn.v_proj")

dev = "cuda" if torch.cuda.is_available() else "hip"
device = torch.device(dev)

with open(os.path.join(MODEL, "model.safetensors.index.json")) as f:
    wmap = json.load(f)["weight_map"]
shard = os.path.join(MODEL, wmap[f"{PREFIX}.trellis"])
with safe_open(shard, framework="pt", device="cpu") as sf:
    trellis = sf.get_tensor(f"{PREFIX}.trellis").to(device).contiguous()
    ic = sf.get_tensor(f"{PREFIX}.suh").numel()
    oc = sf.get_tensor(f"{PREFIX}.svh").numel()
k = trellis.shape[-1] / 16
k = int(k) if float(k).is_integer() else k

print(f"device={dev} torch={torch.__version__} prefix={PREFIX} k={k} "
      f"trellis={tuple(trellis.shape)} ic={ic} oc={oc}", flush=True)

w = torch.empty((ic, oc), dtype=torch.half, device=device)
ext.reconstruct(w, trellis, k, False, False)
w = w.cpu().contiguous()  # copy syncs the stream
torch.save({"w": w, "device": dev, "k": k, "prefix": PREFIX}, OUT)
print(f"saved {OUT} shape={tuple(w.shape)} max={w.abs().max().item():.6f}", flush=True)
