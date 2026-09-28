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
cu = torch.tensor([0, L], device="cuda", dtype=torch.int32)

# A: with random h0
oA, _ = fla(q, k, v, g, beta, initial_state=h0, output_final_state=True, cu_seqlens=cu)
# B: with ZERO h0
h0z = torch.zeros_like(h0)
oB, _ = fla(q, k, v, g, beta, initial_state=h0z, output_final_state=True, cu_seqlens=cu)
# C: no h0
oC, _ = fla(q, k, v, g, beta, output_final_state=True, cu_seqlens=cu)

print(tag, "A(h0) o absmax", oA.float().abs().max().item(),
      "B(zh0) o absmax", oB.float().abs().max().item(),
      "C(no h0) o absmax", oC.float().abs().max().item())
# per-position max diff between A and B (isolates h0 effect)
dAB = (oA.float()-oB.float()).abs()[0].amax(dim=(1,2))
print(tag, "A-B per-pos (h0 effect):", [round(x,3) for x in dAB.tolist()])
dBC = (oB.float()-oC.float()).abs()
print(tag, "B-C max_abs (zero vs none h0):", dBC.max().item())
torch.save({"oA":oA.cpu(),"oB":oB.cpu(),"oC":oC.cpu()}, f"/tmp/fla2_{tag}.pt")
