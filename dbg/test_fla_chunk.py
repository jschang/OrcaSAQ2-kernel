import torch, sys
tag = sys.argv[1] if len(sys.argv) > 1 else "out"
torch.manual_seed(1234)
B, L = 1, 16
Hq, Hv, K, V = 16, 48, 128, 128
q = torch.randn(B, L, Hq, K, device="cuda", dtype=torch.bfloat16)
k = torch.nn.functional.normalize(torch.randn(B, L, Hq, K, device="cuda", dtype=torch.bfloat16), p=2, dim=-1)
v = torch.randn(B, L, Hv, V, device="cuda", dtype=torch.bfloat16)
g = torch.nn.functional.logsigmoid(torch.rand(B, L, Hv, device="cuda", dtype=torch.float32))
beta = torch.rand(B, L, Hv, device="cuda", dtype=torch.bfloat16).sigmoid()
h0 = torch.randn(B, Hv, V, K, device="cuda", dtype=torch.float32)

from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule as fla

# equal-length form
o1, fs1 = fla(q, k, v, g, beta, initial_state=h0, output_final_state=True)

# variable-length form (single seq), as the model uses
cu = torch.tensor([0, L], device="cuda", dtype=torch.int32)
o2, fs2 = fla(q, k, v, g, beta, initial_state=h0, output_final_state=True, cu_seqlens=cu)

print(tag, "o1 finite", torch.isfinite(o1).all().item(), "o1 absmax", o1.float().abs().max().item())
print(tag, "fs1 finite", torch.isfinite(fs1).all().item(), "fs1 absmax", fs1.float().abs().max().item())
print(tag, "o2 finite", torch.isfinite(o2).all().item(), "o2 absmax", o2.float().abs().max().item())
print(tag, "o1[0,0,:4,:4] =")
print(o1[0,0,:4,:4].float().cpu().numpy().round(4))
print(tag, "fs1[0,0,:4,:4] =")
print(fs1[0,0,:4,:4].float().cpu().numpy().round(4))
torch.save({"o1": o1.cpu(), "fs1": fs1.cpu(), "o2": o2.cpu(), "fs2": fs2.cpu()}, f"/tmp/fla_{tag}.pt")
