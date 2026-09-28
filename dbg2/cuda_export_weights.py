"""Export dequantized EXL3 projection weights to FP16, for cross-GPU comparison.

Run INSIDE the reference (CUDA) container -- this uses the CUDA build of the
exllamav3 extension's `ext.reconstruct`, so the output is exactly the weight the
CUDA model serves. It loads each projection's EXL3 tensors straight from the
checkpoint safetensors and runs the plugin's own reconstruct_weight (same code the
server uses), saving {prefix: fp16(out, in)} to a .pt file.

Memory-frugal by design so a live serving container can host it without OOM:
one projection at a time, the GPU tensor is moved to CPU and freed before the next.

Env:
  MODEL_PATH     (default /model)
  OUT            (default /tmp/dequant_ref.pt)
  PROJECTIONS    comma-separated prefixes (default: GDN/attn/MLP of layers 0 and 3)
"""
import json
import os

import torch
from safetensors import safe_open

MODEL = os.environ.get("MODEL_PATH", "/model")
OUT = os.environ.get("OUT", "/tmp/dequant_ref.pt")
DEFAULT_PROJ = ",".join([
    "model.language_model.layers.0.linear_attn.in_proj_qkv",
    "model.language_model.layers.0.linear_attn.out_proj",
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.0.mlp.up_proj",
    "model.language_model.layers.0.mlp.down_proj",
    "model.language_model.layers.3.self_attn.q_proj",
    "model.language_model.layers.3.self_attn.k_proj",
    "model.language_model.layers.3.self_attn.v_proj",
    "model.language_model.layers.3.self_attn.o_proj",
])
PROJECTIONS = [p for p in os.environ.get("PROJECTIONS", DEFAULT_PROJ).split(",") if p]

BLOCK = 128
_HAD = {}


def _had(device, dtype):
    key = (str(device), dtype)
    if key not in _HAD:
        h = torch.tensor([[1.0]])
        while h.shape[0] < BLOCK:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _HAD[key] = (h / BLOCK ** 0.5).to(device=device, dtype=dtype)
    return _HAD[key]


def _t128(x):
    s = x.shape
    return (x.reshape(*s[:-1], s[-1] // BLOCK, BLOCK) @ _had(x.device, x.dtype)).reshape(s)


# Inline copy of orcasaq2/__init__.py::reconstruct_weight (device-agnostic), so this
# script only needs the exllamav3 extension and not the full vLLM plugin.
def reconstruct_weight(ext, trellis, suh, svh, mcg, mul1):
    ic, oc = suh.numel(), svh.numel()
    k = trellis.shape[-1] / 16
    k = int(k) if float(k).is_integer() else k
    w = torch.empty((ic, oc), dtype=torch.half, device=trellis.device)
    ext.reconstruct(w, trellis, k, mcg is not None, mul1 is not None)
    # Free the GPU immediately: everything below runs on CPU so a live serving
    # container never holds a float32 copy of a big projection.
    w = w.cpu().float()
    suh = suh.cpu().float()
    svh = svh.cpu().float()
    w = _t128(w.t()).t() * suh.unsqueeze(1)
    w = _t128(w) * svh.unsqueeze(0)
    return w.t().contiguous().half()


idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))
wm = idx["weight_map"]


def load_tensor(name, device):
    f = os.path.join(MODEL, wm[name])
    with safe_open(f, framework="pt", device="cpu") as sf:
        return sf.get_tensor(name).to(device).contiguous()


from exllamav3.ext import exllamav3_ext as ext  # noqa: E402  (needs torch + GPU first)

dev = torch.device("cuda", 0)
devname = "cuda" if (torch.version.hip is None) else "rocm"
print(f"device={devname} torch={torch.__version__}", flush=True)

result = {}
for prefix in PROJECTIONS:
    names = {}
    for suf in ("trellis", "suh", "svh", "mcg", "mul1"):
        full = f"{prefix}.{suf}"
        if full in wm:
            names[suf] = load_tensor(full, dev)
    if "trellis" not in names:
        print(f"SKIP {prefix}: no .trellis (dense/unquantized)", flush=True)
        continue
    w = reconstruct_weight(ext, names["trellis"], names["suh"], names["svh"],
                           names.get("mcg"), names.get("mul1"))
    result[prefix] = w.detach().cpu().contiguous()
    del w, names
    torch.cuda.empty_cache()
    print(f"OK  {prefix}: out={result[prefix].shape[0]} in={result[prefix].shape[1]} "
          f"max={result[prefix].abs().max().item():.4f}", flush=True)
    # Save incrementally so a later OOM (e.g. a huge lm_head trellis) does not
    # lose the projections already dequantized.
    torch.save({"dequant": result, "device": devname}, OUT)

print(f"saved {len(result)} projections -> {OUT}", flush=True)
