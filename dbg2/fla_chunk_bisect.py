"""Stage-by-stage bisect of FLA chunk_gated_delta_rule on the ROCm stack.

Runs the real vLLM/FLA Triton pipeline stages (cumsum -> kkt -> solve_tril ->
wy_fast -> chunk_delta_h -> chunk_fwd_o) on the live GPU, then compares stages 5
(h/v_new/final_state) and 6 (o) against a torch fp32 reference that transcribes
the kernels' actual computations (bf16 inputs, fp32 accumulate, per-chunk local
cumsum g). T may be partial or full chunks (up to 4 chunks).

usage: python3 fla_chunk_bisect.py [T] [tag] [g_scale] [w_scale] [h0_scale]
        [h0_dtype=fp32|bf16]
"""
import sys
import torch

T = int(sys.argv[1]) if len(sys.argv) > 1 else 64
TAG = sys.argv[2] if len(sys.argv) > 2 else "rocm"
# input scale knobs: 0 disables a term, isolating which part is miscompiled
G_SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
W_SCALE = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
H0_SCALE = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
H0_DTYPE = sys.argv[6] if len(sys.argv) > 6 else "fp32"

from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum
from vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril
from vllm.third_party.flash_linear_attention.ops.wy_fast import recompute_w_u_fwd
from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from vllm.third_party.flash_linear_attention.ops.chunk_o import chunk_fwd_o

BT = FLA_CHUNK_SIZE
assert T <= 4 * BT, "keep it simple: at most four chunks"

torch.manual_seed(1234)
B, Hq, Hv, K, V = 1, 16, 48, 128, 128
q = torch.randn(B, T, Hq, K, device="cuda", dtype=torch.bfloat16)
k = torch.nn.functional.normalize(
    torch.randn(B, T, Hq, K, device="cuda", dtype=torch.bfloat16), p=2, dim=-1
)
v = torch.randn(B, T, Hv, V, device="cuda", dtype=torch.bfloat16)
g_raw = torch.nn.functional.logsigmoid(torch.rand(B, T, Hv, device="cuda", dtype=torch.float32)) * G_SCALE
beta = torch.rand(B, T, Hv, device="cuda", dtype=torch.bfloat16).sigmoid()
h0 = (
    torch.randn(B, Hv, V, K, device="cuda", dtype=torch.float32) * H0_SCALE
).to(torch.bfloat16 if H0_DTYPE == "bf16" else torch.float32)
cu = torch.tensor([0, T], device="cuda", dtype=torch.int32)
scale = K ** -0.5
NT = (T + BT - 1) // BT  # ceil, matches tl.cdiv in the kernels

# ---- run the real pipeline ----
g = chunk_local_cumsum(g_raw, chunk_size=BT, cu_seqlens=cu)
A = chunk_scaled_dot_kkt_fwd(
    k=k, beta=beta, g=g, cu_seqlens=cu, output_dtype=torch.float32
)
A = solve_tril(A=A, cu_seqlens=cu, output_dtype=k.dtype)
w, u = recompute_w_u_fwd(
    k=k, v=v, beta=beta, A=A, g_cumsum=g, cu_seqlens=cu
)
w = w * W_SCALE
h, v_new, ht = chunk_gated_delta_rule_fwd_h(
    k=k, w=w, u=u, g=g, initial_state=h0, output_final_state=True, cu_seqlens=cu
)
o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, cu_seqlens=cu)


# ---- torch fp32 reference (transcribed from the kernels) ----
def ref_stage5():
    h_exp, vnew_list = [], []
    ht_ref = h0[0].float().clone()  # [Hv,V,K]
    for c in range(NT):
        ts, te = c * BT, min((c + 1) * BT, T)
        bt = te - ts
        h_exp.append(ht_ref.to(torch.bfloat16).clone())  # stored before update
        u_c = u[0, ts:te]  # [bt,Hv,V]
        g_c = g[0, ts:te]  # [bt,Hv]
        g_last = g_c[-1]  # [Hv]
        vnew_c = torch.empty_like(u_c)
        vnew_scaled_bf = torch.empty_like(u_c)
        for i_h in range(Hv):
            i_k = i_h // (Hv // Hq)
            w_c = w[0, ts:te, i_h].float()  # [bt,K]
            k_c = k[0, ts:te, i_k].float()  # [bt,K]
            v_raw = u_c[:, i_h].float() - w_c @ ht_ref[i_h].float().T  # [bt,V]
            # stored v_new is the UNSCALED residual (gate applied only for state)
            vnew_c[:, i_h] = v_raw
            vnew_scaled_bf[:, i_h] = (
                v_raw * torch.exp(g_last[i_h] - g_c[:, i_h])[:, None]
            ).to(torch.bfloat16)
        vnew_list.append(vnew_c.to(torch.bfloat16))
        for i_h in range(Hv):
            i_k = i_h // (Hv // Hq)
            k_c = k[0, ts:te, i_k].float()
            if bt < BT:  # zero-pad invalid rows before the state dot
                k_pad = torch.zeros(BT, K, device=k_c.device, dtype=k_c.dtype)
                k_pad[:bt] = k_c
                v_pad = torch.zeros(BT, V, device=vnew_scaled_bf.device, dtype=torch.float32)
                v_pad[:bt] = vnew_scaled_bf[:, i_h].float()
                update = (k_pad.T @ v_pad).T
            else:
                update = (k_c.T @ vnew_scaled_bf[:, i_h].float()).T
            ht_ref[i_h] = ht_ref[i_h] * torch.exp(g_last[i_h]) + update
    return (
        torch.stack(h_exp),
        torch.cat(vnew_list, dim=0),
        ht_ref,
    )


h_ref, vnew_ref, ht_ref = ref_stage5()

o_ref = torch.zeros(B, T, Hv, V, dtype=torch.float32, device="cuda")
for c in range(NT):
    ts, te = c * BT, min((c + 1) * BT, T)
    q_c = q[0, ts:te].float()  # [bt,K]
    k_c = k[0, ts:te].float()  # [bt,K]
    g_c = g[0, ts:te]  # [bt,Hv]
    v_c = v_new[0, ts:te]  # [bt,Hv,V] bf16 (what the kernel actually reads)
    for i_h in range(Hv):
        i_k = i_h // (Hv // Hq)
        h_c = h[0, c, i_h].float()  # [V,K] as stored (bf16->fp32)
        o_inter = (q_c[:, i_k] @ h_c.T) * torch.exp(g_c[:, i_h])[:, None]  # [bt,V]
        A_intra = q_c[:, i_k] @ k_c[:, i_k].T  # [bt,bt]
        A_intra = A_intra * torch.exp(g_c[:, i_h][:, None] - g_c[:, i_h][None, :])
        A_intra = torch.tril(A_intra)
        o_intra = A_intra.to(torch.bfloat16) @ v_c[:, i_h]  # [bt,V] fp32 acc
        o_ref[0, ts:te, i_h] = scale * (o_inter.float() + o_intra.float())


def report(name, got, ref, rtol=2e-2, atol=2e-2):
    d = (got.float() - ref.float()).abs()
    rel = d / ref.float().abs().clamp_min(1e-3)
    bad = ((d > atol) & (rel > rtol)).float().mean().item()
    print(
        f"{name:<12} max_abs {d.max().item():.5f}  mean {d.mean().item():.6f}  "
        f"frac_bad {bad * 100:.2f}%"
    )


report("h[0]", h[0, 0], h_ref[0])
report("h[last]", h[0, NT - 1], h_ref[NT - 1])
report("v_new", v_new[0], vnew_ref)
report("final_st", ht[0], ht_ref[None])
report("o", o[0], o_ref[0])

# bit-exact check on the first stored chunk state (pure load+store, no dot)
bit_ok = torch.equal(h[0, 0].to(torch.float32), h0[0].to(torch.bfloat16).to(torch.float32))
print("h[0]==h0(bf16) bitwise:", bit_ok)
torch.save(
    {
        "g": g, "A": A, "w": w, "u": u, "h": h, "v_new": v_new, "ht": ht, "o": o,
        "h_ref": h_ref, "vnew_ref": vnew_ref, "ht_ref": ht_ref, "o_ref": o_ref,
        "h0": h0,
    },
    f"/tmp/fla_bisect_{TAG}.pt",
)
print("saved /tmp/fla_bisect_%s.pt" % TAG)
