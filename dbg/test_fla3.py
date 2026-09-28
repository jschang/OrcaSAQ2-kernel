import torch, sys
tag = sys.argv[1] if len(sys.argv) > 1 else "out"
torch.manual_seed(7)
B = 1
HV, V, K = 48, 128, 128
qk_dim = 2 * 16 * 128
qkv_dim = qk_dim + HV * V
mixed_qkv = torch.randn(B, qkv_dim, device="cuda", dtype=torch.bfloat16)
a = torch.randn(B, HV, device="cuda", dtype=torch.float32)
b = torch.randn(B, HV, device="cuda", dtype=torch.float32)
A_log = torch.randn(HV, device="cuda", dtype=torch.float32)
dt_bias = torch.randn(HV, device="cuda", dtype=torch.float32)
initial_state = torch.randn(4, HV, V, K, device="cuda", dtype=torch.float32)
state_before = initial_state.clone()
out = torch.zeros(B, 1, HV, V, device="cuda", dtype=torch.bfloat16)
ssm_idx = torch.tensor([0], device="cuda", dtype=torch.int32)
scale = K ** -0.5

from vllm.third_party.flash_linear_attention.ops import fused_recurrent_gated_delta_rule_packed_decode as fr
for l2n in (True, False):
    out.zero_()
    st = initial_state.clone()
    fr(mixed_qkv, a, b, A_log, dt_bias, scale, st, out, ssm_idx, use_qk_l2norm_in_kernel=l2n)
    dstate = (st - state_before).abs().max().item()
    print(f"{tag} l2n={l2n}: out_absmax={out.float().abs().max().item():.4f} state_changed_max={dstate:.4f}")
    print(f"    out[0,0,:3,:3]={out[0,0,:3,:3].float().tolist()}")
print(tag, "state_before_absmax", state_before.abs().max().item())
