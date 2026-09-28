import torch, sys
tag = sys.argv[1] if len(sys.argv) > 1 else "out"
torch.manual_seed(11)
L = 16
qkv_dim = 2*16*128 + 48*128   # [q,k,v] packed = 10240
conv_output = torch.randn(L, qkv_dim, device="cuda", dtype=torch.bfloat16)
a = torch.randn(L, 48, device="cuda", dtype=torch.bfloat16)
b = torch.randn(L, 48, device="cuda", dtype=torch.bfloat16)
A_log = torch.randn(48, device="cuda", dtype=torch.float32)
dt_bias = torch.randn(48, device="cuda", dtype=torch.float32)

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import fused_post_conv_prep
q,k,v,g,beta = fused_post_conv_prep(conv_output, a, b, A_log, dt_bias, 16, 128, 128, apply_l2norm=True, output_g_exp=False)
for name,t in [("q",q),("k",k),("v",v),("g",g),("beta",beta)]:
    print(f"{tag} {name}: shape={tuple(t.shape)} dtype={t.dtype} absmax={t.float().abs().max().item():.5f}")
torch.save({"q":q.cpu(),"k":k.cpu(),"v":v.cpu(),"g":g.cpu(),"beta":beta.cpu()}, f"/tmp/fla4_{tag}.pt")
