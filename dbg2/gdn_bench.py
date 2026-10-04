"""#8 probe v3: kernel-only CUDA time (torch.profiler) for FLA vs aiter GDN decode."""
import torch

torch.manual_seed(0)
dev = "cuda"
HK, HV, DK, DV = 16, 48, 128, 128
KEY_DIM, VAL_DIM = HK * DK, HV * DV
B = 1

mixed_qkv_2d = torch.randn(B, 2 * KEY_DIM + VAL_DIM, dtype=torch.bfloat16, device=dev) * 0.1
mixed_qkv_3d = mixed_qkv_2d.unsqueeze(-1)
a = torch.randn(B, HV, dtype=torch.bfloat16, device=dev) * 0.1
b_gate = torch.randn(B, HV, dtype=torch.bfloat16, device=dev) * 0.1
A_log = torch.randn(HV, dtype=torch.float32, device=dev) * 0.5
dt_bias = torch.randn(HV, dtype=torch.bfloat16, device=dev) * 0.1
idx = torch.tensor([0], dtype=torch.int32, device=dev)

from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from aiter.ops.fused_split_gdr_update import fused_split_gdr_update

fla_state = torch.zeros(1, HV, DK, DV, dtype=torch.float32, device=dev)
fla_out = torch.empty(B, 1, HV, DV, dtype=torch.bfloat16, device=dev)
aiter_state = torch.zeros(1, HV, DK // 4, DV, 4, dtype=torch.float32, device=dev)

def fla_step():
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv_2d, a=a, b=b_gate, A_log=A_log, dt_bias=dt_bias,
        scale=DK ** -0.5, initial_state=fla_state, out=fla_out,
        ssm_state_indices=idx, use_qk_l2norm_in_kernel=True,
    )

def aiter_step():
    fused_split_gdr_update(
        mixed_qkv_3d, A_log, a, dt_bias, b_gate,
        aiter_state, idx, KEY_DIM, VAL_DIM, HK, HV, DK,
    )

def profile_kernels(fn, label, iters=200):
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total_us = 0.0
    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.self_device_time_total > 0:
            print(f"  {label}: {evt.key[:70]:70s} {evt.self_device_time_total/iters:8.2f} us/call  n={evt.count}")
            total_us += evt.self_device_time_total / iters
    print(f"  {label}: TOTAL kernel GPU {total_us:8.2f} us/call")

profile_kernels(fla_step, "FLA")
profile_kernels(aiter_step, "aiter")
