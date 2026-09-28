"""Compare the ROCm reconstruct_weight output against a CUDA-exported reference.

Run INSIDE the ROCm container. For each projection it loads the EXL3 tensors from the
checkpoint, runs the ROCm build of reconstruct_weight, loads the CUDA reference .pt
(produced by cuda_export_weights.py), and reports:

  * overall cosine, max/mean abs diff
  * per-row and per-col cosine (min / median / fraction > 0.99) -> partial-bug detection
  * orientation: cos(a,b) vs cos(a,-b) -> global sign flip?
  * elementwise exact-match fraction (rounded) -> identical vs genuinely different

Reading the result:
  cos~1, max~0        -> dequant identical on CUDA and ROCm (dequant is NOT the bug)
  cos(a,-b)~1         -> global sign flip (one of the two is negated)
  per-row min low     -> only some rows wrong (partial index/permutation bug)
  cos low both ways   -> genuinely different weights (ROCm ext.reconstruct differs)

Env:
  MODEL_PATH   (default /model)
  REF          (default /dbg2/cuda_dequant_ref.pt)
  PROJECTIONS  comma-separated prefixes (default matches cuda_export_weights.py)
"""
import json
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL = os.environ.get("MODEL_PATH", "/model")
REF = os.environ.get("REF", "/dbg2/cuda_dequant_ref.pt")
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


def reconstruct_weight(ext, trellis, suh, svh, mcg, mul1):
    ic, oc = suh.numel(), svh.numel()
    k = trellis.shape[-1] / 16
    k = int(k) if float(k).is_integer() else k
    w = torch.empty((ic, oc), dtype=torch.half, device=trellis.device)
    ext.reconstruct(w, trellis, k, mcg is not None, mul1 is not None)
    # Free the GPU immediately; the Hadamard/scaling and comparison run on CPU.
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


from exllamav3.ext import exllamav3_ext as ext  # noqa: E402

devname = "cuda" if (torch.version.hip is None) else "rocm"
print(f"local device={devname} torch={torch.__version__}", flush=True)

ref = torch.load(REF, map_location="cpu")
print(f"reference device={ref.get('device')} n={len(ref['dequant'])}", flush=True)
dev = torch.device("cuda", 0)

for prefix in PROJECTIONS:
    names = {}
    for suf in ("trellis", "suh", "svh", "mcg", "mul1"):
        full = f"{prefix}.{suf}"
        if full in wm:
            names[suf] = load_tensor(full, dev)
    if "trellis" not in names or prefix not in ref["dequant"]:
        print(f"SKIP {prefix}", flush=True)
        continue
    a = reconstruct_weight(ext, names["trellis"], names["suh"], names["svh"],
                           names.get("mcg"), names.get("mul1")).float()
    b = ref["dequant"][prefix].float()
    del names
    torch.cuda.empty_cache()

    d = (a - b).abs()
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    cos_neg = F.cosine_similarity(a.flatten(), b.flatten().neg(), dim=0).item()
    rowcos = F.cosine_similarity(a, b, dim=1)
    colcos = F.cosine_similarity(a, b, dim=0)
    ar, br = torch.round(a, decimals=4), torch.round(b, decimals=4)
    exact = (ar == br).float().mean().item()

    print(f"== {prefix}  {tuple(a.shape)}", flush=True)
    print(f"   cos={cos:.6f}  cos_neg={cos_neg:.6f}  "
          f"max_abs={d.max().item():.6f}  mean_abs={d.mean().item():.6f}  "
          f"exact4={exact:.4f}", flush=True)
    print(f"   row cos: min={rowcos.min().item():.4f} med={rowcos.median().item():.4f} "
          f"frac>0.99={ (rowcos > 0.99).float().mean().item():.3f}", flush=True)
    print(f"   col cos: min={colcos.min().item():.4f} med={colcos.median().item():.4f} "
          f"frac>0.99={ (colcos > 0.99).float().mean().item():.3f}", flush=True)
