import torch
from vllm.third_party.flash_linear_attention.ops.chunk import l2norm_fwd

torch.manual_seed(0)
# x: [B, L, H, D] or any shape; l2norm along last dim
x = torch.randn(2, 16, 4, 64, device="cuda", dtype=torch.bfloat16)

out = l2norm_fwd(x, eps=1e-6)

# reference
ref = x.float() / (torch.linalg.norm(x.float(), dim=-1, keepdim=True) + 1e-6)
ref = ref.to(x.dtype)

diff = (out.float() - ref.float()).abs()
print("l2norm: out.shape", tuple(out.shape), "max_abs_diff", diff.max().item(),
      "mean_abs_diff", diff.mean().item())
print("out finite:", torch.isfinite(out).all().item())

# also test fp16 / fp32
for dt in (torch.float16, torch.float32):
    xs = torch.randn(2, 16, 4, 64, device="cuda", dtype=dt)
    os_ = l2norm_fwd(xs, eps=1e-6)
    rs = xs / (torch.linalg.norm(xs, dim=-1, keepdim=True) + 1e-6)
    print(f"l2norm {dt}: max_abs_diff {(os_-rs).abs().max().item():.6e}")
