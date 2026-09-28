import json
import os

import torch
from safetensors import safe_open

from exllamav3.ext import exllamav3_ext as ext


MODEL_PATH = os.environ.get("MODEL_PATH", "/model")
ROWS = int(os.environ.get("ROWS", "4"))


with open(os.path.join(MODEL_PATH, "model.safetensors.index.json")) as f:
    IDX = json.load(f)["weight_map"]


def load_tensor(key, device="cuda"):
    with safe_open(os.path.join(MODEL_PATH, IDX[key]), framework="pt", device="cpu") as f:
        tensor = f.get_tensor(key)
    return tensor.to(device)


def main():
    torch.manual_seed(1234)
    trellis = load_tensor("lm_head.trellis").contiguous()
    suh = load_tensor("lm_head.suh").to(torch.float16).contiguous()
    svh = load_tensor("lm_head.svh").to(torch.float16).contiguous()
    mul1 = "lm_head.mul1" in IDX
    mcg = "lm_head.mcg" in IDX
    k = trellis.shape[-1] / 16
    k = int(k) if float(k).is_integer() else k
    ic = suh.numel()
    oc = svh.numel()
    print("lm_head", "trellis", tuple(trellis.shape), "ic", ic, "oc", oc, "K", k)

    x = (torch.randn(ROWS, ic, device="cuda", dtype=torch.float16) * 0.2).contiguous()

    fused = torch.empty((ROWS, oc), dtype=torch.float16, device="cuda")
    ext.exl3_gemm(
        x,
        trellis,
        fused,
        suh,
        torch.empty_like(x),
        svh,
        -1,
        mcg,
        mul1,
        0,
    )

    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)
    w = torch.empty((ic, oc), dtype=torch.float16, device="cuda")
    ext.reconstruct(w, trellis, k, mcg, mul1)
    ref = torch.empty((ROWS, oc), dtype=torch.float16, device="cuda")
    ext.hgemm_recon(xh, w, ref)
    ext.had_r_128(ref, ref, None, svh, 1.0)
    torch.cuda.synchronize()

    d = (fused.detach().cpu().float() - ref.detach().cpu().float()).abs()
    print(
        "fused_vs_reconstruct",
        "max_abs",
        float(d.max()),
        "mean_abs",
        float(d.mean()),
        "bad_gt_1e-2",
        int((d > 1e-2).sum()),
        "numel",
        d.numel(),
    )
    print("top fused", torch.topk(fused[0].float().cpu(), 8).indices.tolist())
    print("top ref   ", torch.topk(ref[0].float().cpu(), 8).indices.tolist())


if __name__ == "__main__":
    main()
