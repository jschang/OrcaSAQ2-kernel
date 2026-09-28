"""Stage-by-stage isolation for the oracle.

1. Per-layer state norms from the saved oracle run (localize the bad layer).
2. Tokenization check.
3. Layer-0 GDN cross-checks: conv vs F.conv1d, recurrence vs FLA chunk kernel,
   gating values.
"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/dbg2")
import orca_oracle as oo

DEV = oo.DEV


def part1_saved_states():
    d = torch.load(f"{oo.OUT_DIR}/oracle_layer_states.pt", map_location="cpu")
    ls = d["layer_states"]
    print("=== saved oracle: prompt/id ===")
    print(d["prompt"], d["token_ids"])
    print("=== per-layer last-row L2 norms ===")
    print(f"embed: {ls['embed'][:, -1].float().norm().item():.4f}")
    prev = None
    for i in sorted(k for k in ls if isinstance(k, int)):
        n = ls[i][:, -1].float().norm().item()
        flag = ""
        if prev is not None and (n > 3 * prev or n < prev / 3):
            flag = "  <<< jump"
        print(f"layer {i:>2} ({oo.LAYER_TYPES[i]:<16}): {n:12.4f}{flag}")
        prev = n
    print(f"final_hidden: {ls['final_hidden'][:, -1].float().norm().item():.4f}")


def part2_tokens():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(oo.MODEL_PATH)
    ids = tok("The capital of France is", add_special_tokens=False)["input_ids"]
    print("=== tokens ===")
    print("ids:", ids)
    print("decoded:", [tok.decode([i]) for i in ids])


def part3_gdn_crosscheck():
    torch.cuda.set_device(0)
    emb_q = oo.load_tensor("model.language_model.embed_tokens.qweight")
    emb_s = oo.load_tensor("model.language_model.embed_tokens.scales").to(oo.BF16)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(oo.MODEL_PATH)
    ids = torch.tensor(
        tok("The capital of France is", add_special_tokens=False)["input_ids"], device=DEV)
    emb = emb_q[ids].to(oo.BF16) * emb_s[ids].view(-1, 1)
    in_ln = oo.load_tensor("model.language_model.layers.0.input_layernorm.weight")
    h = oo.gemma_rms(emb, in_ln)
    T = h.shape[0]
    W = oo.GdnLayerW(0)

    qkv = oo.exl3_apply(h, W.qkv)
    z = oo.exl3_apply(h, W.z)
    b = F.linear(h, W.w_b)
    a = F.linear(h, W.w_a)

    # --- conv: mine vs F.conv1d (independent) ---
    D = qkv.shape[1]
    acc = torch.zeros(T, D, dtype=torch.float32, device=DEV)
    for j in range(oo.CONV_W):
        wj = W.conv[:, j].to(oo.BF16)
        if j == 0:
            xt = qkv
        else:
            xt = torch.cat([torch.zeros(j, D, dtype=oo.BF16, device=DEV), qkv[:T - j]])
        acc += (xt * wj).float()
    qkv_c = F.silu(acc).to(oo.BF16)

    Wrev = W.conv.flip(1).unsqueeze(1)
    pad = oo.CONV_W - 1
    xp = torch.cat([torch.zeros(pad, D, device=DEV), qkv], dim=0).float().transpose(0, 1).unsqueeze(0)
    y = F.conv1d(xp, Wrev, groups=D)
    ref = y[0].t().contiguous()
    ref_c = F.silu(ref).to(oo.BF16)
    print("=== conv diff (mine vs F.conv1d) ===")
    print("max abs diff:", (qkv_c.float() - ref_c.float()).abs().max().item())

    q = qkv_c[:, :oo.K_DIM * oo.K_HEADS].view(T, oo.K_HEADS, oo.K_DIM).float()
    k = qkv_c[:, oo.K_DIM * oo.K_HEADS:2 * oo.K_DIM * oo.K_HEADS].view(T, oo.K_HEADS, oo.K_DIM).float()
    v = qkv_c[:, 2 * oo.K_DIM * oo.K_HEADS:].view(T, oo.V_HEADS, oo.V_DIM)
    q = (q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6)).to(oo.BF16)
    k = (k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)).to(oo.BF16)
    g = -W.A_log.exp() * F.softplus(a.float() + W.dt_bias)
    beta = torch.sigmoid(b.float())
    print("g row0[:4]:", g[0, :4].tolist())
    print("beta row0[:4]:", beta[0, :4].tolist())
    print("q norm row0 head0:", q[0, 0].norm().item())
    print("k norm row0 head0:", k[0, 0].norm().item())
    print("v row0 head0 norm:", v[0, 0].float().norm().item())

    # --- recurrence: mine vs FLA chunk (independent) ---
    G = oo.V_HEADS // oo.K_HEADS
    qf = (q.float() * (oo.K_DIM ** -0.5)).repeat_interleave(G, dim=1)
    kf = k.float().repeat_interleave(G, dim=1)
    vf = v.float()
    S = torch.zeros(oo.V_HEADS, oo.K_DIM, oo.V_DIM, dtype=torch.float32, device=DEV)
    o_naive = torch.empty(T, oo.V_HEADS, oo.V_DIM, dtype=torch.float32, device=DEV)
    for t in range(T):
        S = S * torch.exp(g[t]).view(oo.V_HEADS, 1, 1)
        v_new = vf[t] - torch.bmm(kf[t].unsqueeze(1), S).squeeze(1)
        v_new = v_new * beta[t].view(oo.V_HEADS, 1)
        S = S + v_new.unsqueeze(2) * kf[t].unsqueeze(-1)
        o_naive[t] = torch.einsum("hkv,hv->hk", S, qf[t])

    from vllm.third_party.flash_linear_attention.ops.chunk import (
        chunk_gated_delta_rule,
    )
    o_fla, _ = chunk_gated_delta_rule(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0), beta.unsqueeze(0))
    o_fla = o_fla.squeeze(0)
    d = (o_fla.float() - o_naive).abs()
    print("=== GDN recurrence diff (mine vs FLA chunk) ===")
    print("max abs:", d.max().item(), " mean:", d.mean().item())
    print("o_naive row0 head0 [:8]:", o_naive[0, 0, :8].tolist())
    print("o_fla   row0 head0 [:8]:", o_fla[0, 0, :8].tolist())

    # --- full oracle layer-0 output vs recomputed continuation ---
    # (continuation not needed here; norms in part1 localize)




def part4_gemm_consistency():
    # For each unverified shard type, G(I) gives the dequantized weight row-major
    # (G(x) = x @ Wdeq^T), so G(x) must equal x @ G(I) for random x.
    torch.cuda.set_device(0)
    bases = [
        f"model.language_model.layers.0.linear_attn.in_proj_qkv",
        f"model.language_model.layers.0.linear_attn.in_proj_z",
        f"model.language_model.layers.0.linear_attn.out_proj",
        f"model.language_model.layers.0.mlp.gate_proj",
        f"model.language_model.layers.0.mlp.up_proj",
        f"model.language_model.layers.0.mlp.down_proj",
        f"model.language_model.layers.3.self_attn.q_proj",
        f"model.language_model.layers.3.self_attn.k_proj",
        f"model.language_model.layers.3.self_attn.v_proj",
        f"model.language_model.layers.3.self_attn.o_proj",
        "lm_head",
    ]
    torch.manual_seed(1)
    expect = {
        "in_proj_qkv": (5120, 10240), "in_proj_z": (5120, 6144), "out_proj": (6144, 5120),
        "gate_proj": (5120, 17408), "up_proj": (5120, 17408), "down_proj": (17408, 5120),
        "q_proj": (5120, 12288), "k_proj": (5120, 1024), "v_proj": (5120, 1024),
        "o_proj": (6144, 5120), "lm_head": (5120, 248320),
    }
    for base in bases:
        s = oo.load_shard(base)
        K = s["k"]
        name = base.split(".")[-1]
        assert (K, s["oc"]) == expect[name], (name, K, s["oc"], expect[name])
        I = torch.eye(K, dtype=torch.float16, device=DEV)
        Wt = oo.exl3_apply(I, s)  # [K, oc] fp16 -> bf16
        x = torch.randn(4, K, dtype=torch.bfloat16, device=DEV)
        y1 = oo.exl3_apply(x, s).half()
        y2 = x.half() @ Wt
        rel = (y1.float() - y2.float()).norm().item() / max(y2.float().norm().item(), 1e-9)
        print(f"{name:<12} K={K:<6} oc={s['oc']:<6} mcg={s['mcg']} mul1={s['mul1']}  rel diff: {rel:.3e}")
        del I, Wt, x, y1, y2, s
        torch.cuda.empty_cache()

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "1"):
        part1_saved_states()
    if which in ("all", "2"):
        part2_tokens()
    if which in ("all", "3"):
        part3_gdn_crosscheck()
    if which in ("all", "4"):
        part4_gemm_consistency()
