"""Full-model oracle: the exact served EXL3 weights, plain torch math, no runner.

Purpose: discriminate between "weights+math" and "ROCm runner".
- EXL3 GEMM: same exllamav3 kernel as serving (verified byte-identical on this GPU).
- Everything else: pure-torch reference math (fp32 causal conv, l2norm, gating,
  naive per-token GDN recurrence in fp32, bf16 SDPA, GemmaRMSNorm in fp32).
- No attention metadata, no block tables, no ssm/conv state (fresh zero state).

References:
  CUDA server: top-1 " Paris" @ -0.4135
  ROCm server: top-1 " the"   @ -1.27

Usage: python /dbg2/orca_oracle.py   (env: MODEL_PATH=/model, OUT_DIR=/dbg2)
"""
import json
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open
from exllamav3.ext import exllamav3_ext as ext

MODEL_PATH = os.environ.get("MODEL_PATH", "/model")
OUT_DIR = os.environ.get("OUT_DIR", "/dbg2")
DEV = torch.device("cuda", 0)
PROMPT = os.environ.get("ORACLE_PROMPT", "The capital of France is")
TOPK = int(os.environ.get("ORACLE_TOPK", "10"))
DENSE = os.environ.get("ORACLE_DENSE", "0") == "1"
SAVE_INT = os.environ.get("ORACLE_SAVE_INT", "1") == "1"

IDX = json.load(open(os.path.join(MODEL_PATH, "model.safetensors.index.json")))["weight_map"]
CFG = json.load(open(os.path.join(MODEL_PATH, "config.json")))
TC = CFG["text_config"]

HIDDEN = TC["hidden_size"]
NLAYERS = TC["num_hidden_layers"]
LAYER_TYPES = TC["layer_types"]
N_HEADS = TC["num_attention_heads"]
N_KV = TC["num_key_value_heads"]
HEAD_DIM = TC["head_dim"]
ATTN_SCALE = HEAD_DIM ** -0.5
ROPE = TC["rope_parameters"]
ROTARY_DIM = int(HEAD_DIM * ROPE["partial_rotary_factor"])
ROPE_BASE = float(ROPE["rope_theta"])
K_HEADS = TC["linear_num_key_heads"]
V_HEADS = TC["linear_num_value_heads"]
K_DIM = TC["linear_key_head_dim"]
V_DIM = TC["linear_value_head_dim"]
CONV_W = TC["linear_conv_kernel_dim"]
INTER = TC["intermediate_size"]
EPS = float(TC["rms_norm_eps"])
VOCAB = TC["vocab_size"]
BF16 = torch.bfloat16

print(f"config: hidden={HIDDEN} layers={NLAYERS} heads={N_HEADS} kv={N_KV} "
      f"head_dim={HEAD_DIM} rotary={ROTARY_DIM} base={ROPE_BASE} "
      f"k_heads={K_HEADS} v_heads={V_HEADS} kv_dim={K_DIM}/{V_DIM} conv={CONV_W} "
      f"inter={INTER} eps={EPS} vocab={VOCAB}", flush=True)
print(f"gdn layers: {sum(1 for t in LAYER_TYPES if t == 'linear_attention')} "
      f"full layers: {sum(1 for t in LAYER_TYPES if t == 'full_attention')}", flush=True)


def load_tensor(key, dtype=None):
    with safe_open(os.path.join(MODEL_PATH, IDX[key]), framework="pt", device="cpu") as f:
        t = f.get_tensor(key)
    if dtype is not None:
        t = t.to(dtype)
    return t.to(DEV)


def load_shard(base):
    trellis = load_tensor(base + ".trellis").contiguous()
    suh = load_tensor(base + ".suh", torch.float16).contiguous()
    svh = load_tensor(base + ".svh", torch.float16).contiguous()
    k = int(trellis.shape[0] * 16)  # trellis layout: [K/16, oc/16, 48]
    has_mcg = base + ".mcg" in IDX
    has_mul1 = base + ".mul1" in IDX
    if DENSE:
        mcg = load_tensor(base + ".mcg") if has_mcg else None
        mul1 = load_tensor(base + ".mul1") if has_mul1 else None
        return {"trellis": trellis, "suh": suh, "svh": svh, "oc": svh.numel(),
                "k": k, "mcg": mcg, "mul1": mul1}
    return {"trellis": trellis, "suh": suh, "svh": svh, "oc": svh.numel(),
            "k": k, "mcg": has_mcg, "mul1": has_mul1}


def exl3_apply(x, s):
    rows = x.numel() // x.shape[-1]
    if DENSE:
        if "dense" not in s:
            from orcasaq2 import reconstruct_weight
            s["dense"] = reconstruct_weight(
                s["trellis"], s["suh"], s["svh"], s["mcg"], s["mul1"])
            del s["trellis"]
        w = s["dense"]  # (out, in) fp16
        xf = x.reshape(rows, x.shape[-1]).float()
        y = xf @ w.float().t()  # (rows, out) fp32
        return y.to(x.dtype).reshape(*x.shape[:-1], s["oc"])
    xf = x.reshape(rows, x.shape[-1]).half().contiguous()
    y = torch.empty((rows, s["oc"]), dtype=torch.float16, device=DEV)
    ext.exl3_gemm(xf, s["trellis"], y, s["suh"], torch.empty_like(xf), s["svh"], -1,
                  s["mcg"], s["mul1"], 0)
    return y.to(x.dtype).reshape(*x.shape[:-1], s["oc"])


def gemma_rms(x, w):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return (y * (w.float() + 1.0)).to(x.dtype)


def _free_layer(W):
    """Release dense/EXL3 shard buffers for a processed layer (DENSE mode memory)."""
    if W is None:
        return
    for v in vars(W).values():
        if isinstance(v, dict):
            for kk in ("dense", "trellis", "suh", "svh", "mcg", "mul1"):
                v.pop(kk, None)


_inv_freq = 1.0 / (ROPE_BASE ** (
    torch.arange(0, ROTARY_DIM, 2, device=DEV, dtype=torch.float32) / ROTARY_DIM))


def apply_rope(x, cos, sin):
    # x: [T, H, HEAD_DIM] bf16; cos/sin: [T, 1, ROTARY_DIM//2] bf16
    xp = x[..., ROTARY_DIM:]
    xr = x[..., :ROTARY_DIM]
    x1, x2 = torch.chunk(xr, 2, dim=-1)
    yr = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return torch.cat((yr, xp), dim=-1)


class GdnLayerW:
    def __init__(self, i):
        p = f"model.language_model.layers.{i}.linear_attn"
        self.qkv = load_shard(p + ".in_proj_qkv")
        self.z = load_shard(p + ".in_proj_z")
        self.out = load_shard(p + ".out_proj")
        self.w_b = load_tensor(p + ".in_proj_b.weight", BF16)
        self.w_a = load_tensor(p + ".in_proj_a.weight", BF16)
        D = 2 * K_DIM * K_HEADS + V_HEADS * V_DIM
        self.conv = load_tensor(p + ".conv1d.weight").reshape(D, CONV_W).float()
        self.A_log = load_tensor(p + ".A_log").float()
        self.dt_bias = load_tensor(p + ".dt_bias").float()
        self.norm_w = load_tensor(p + ".norm.weight")


def gdn_layer(h, W):
    T = h.shape[0]
    qkv = exl3_apply(h, W.qkv)                # [T, 10240]
    z = exl3_apply(h, W.z)                    # [T, 6144]
    b = F.linear(h, W.w_b)                    # [T, 48]
    a = F.linear(h, W.w_a)                    # [T, 48]

    D = qkv.shape[1]
    acc = torch.zeros(T, D, dtype=torch.float32, device=DEV)
    # vLLM causal_conv1d: weight tap j multiplies x[t-(CONV_W-1-j)]  (w[last]=current)
    for j in range(CONV_W):
        wj = W.conv[:, j].to(BF16)
        offset = CONV_W - 1 - j
        if offset == 0:
            xt = qkv
        else:
            xt = torch.cat([torch.zeros(offset, D, dtype=BF16, device=DEV), qkv[:T - offset]])
        acc += (xt * wj).float()
    qkv_c = F.silu(acc).to(BF16)

    q = qkv_c[:, :K_DIM * K_HEADS].view(T, K_HEADS, K_DIM).float()
    k = qkv_c[:, K_DIM * K_HEADS:2 * K_DIM * K_HEADS].view(T, K_HEADS, K_DIM).float()
    v = qkv_c[:, 2 * K_DIM * K_HEADS:].view(T, V_HEADS, V_DIM)
    q = (q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6)).to(BF16)
    k = (k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)).to(BF16)
    g = -W.A_log.exp() * F.softplus(a.float() + W.dt_bias)   # [T, 48] fp32 log-space
    beta = torch.sigmoid(b.float())                         # [T, 48] fp32

    G = V_HEADS // K_HEADS
    qf = (q.float() * (K_DIM ** -0.5)).repeat_interleave(G, dim=1)
    kf = k.float().repeat_interleave(G, dim=1)
    vf = v.float()
    S = torch.zeros(V_HEADS, K_DIM, V_DIM, dtype=torch.float32, device=DEV)
    o = torch.empty(T, V_HEADS, V_DIM, dtype=torch.float32, device=DEV)
    for t in range(T):
        S = S * torch.exp(g[t]).view(V_HEADS, 1, 1)
        v_new = vf[t] - torch.bmm(kf[t].unsqueeze(1), S).squeeze(1)
        v_new = v_new * beta[t].view(V_HEADS, 1)
        S = S + kf[t].unsqueeze(2) * v_new.unsqueeze(1)
        o[t] = torch.einsum("hkv,hk->hv", S, qf[t])
    core = o.to(BF16)

    x = core.reshape(-1, V_DIM).float()
    zf = z.reshape(-1, V_DIM).float()
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    out = (xn * W.norm_w.float() * F.silu(zf)).to(BF16).view(T, V_HEADS * V_DIM)
    return exl3_apply(out, W.out)


class AttnLayerW:
    def __init__(self, i):
        p = f"model.language_model.layers.{i}.self_attn"
        self.q = load_shard(p + ".q_proj")
        self.k = load_shard(p + ".k_proj")
        self.v = load_shard(p + ".v_proj")
        self.o = load_shard(p + ".o_proj")
        self.q_norm = load_tensor(p + ".q_norm.weight")
        self.k_norm = load_tensor(p + ".k_norm.weight")


def full_attn_layer(h, W, positions):
    T = h.shape[0]
    q_gate = exl3_apply(h, W.q)
    k = exl3_apply(h, W.k)
    v = exl3_apply(h, W.v)
    q_gate = q_gate.view(T, N_HEADS, 2 * HEAD_DIM)
    q, gate = torch.chunk(q_gate, 2, dim=-1)
    q = gemma_rms(q, W.q_norm).view(T, N_HEADS * HEAD_DIM)
    k = gemma_rms(k.view(T, N_KV, HEAD_DIM), W.k_norm).view(T, N_KV * HEAD_DIM)
    freqs = torch.einsum("i,j->ij", positions.float(), _inv_freq)
    cos = freqs.cos().to(BF16).unsqueeze(1)
    sin = freqs.sin().to(BF16).unsqueeze(1)
    q = apply_rope(q.view(T, N_HEADS, HEAD_DIM), cos, sin)
    k = apply_rope(k.view(T, N_KV, HEAD_DIM), cos, sin)
    m = N_HEADS // N_KV
    k = k.repeat_interleave(m, dim=1)
    v = v.view(T, N_KV, HEAD_DIM).repeat_interleave(m, dim=1)
    attn = F.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=True, scale=ATTN_SCALE)
    attn = attn.squeeze(0).reshape(T, N_HEADS * HEAD_DIM).to(BF16)
    attn = attn * torch.sigmoid(gate.reshape(T, N_HEADS * HEAD_DIM))
    return exl3_apply(attn, W.o)


class MlpW:
    def __init__(self, i):
        p = f"model.language_model.layers.{i}.mlp"
        self.gate = load_shard(p + ".gate_proj")
        self.up = load_shard(p + ".up_proj")
        self.down = load_shard(p + ".down_proj")


def mlp_layer(h, W):
    return exl3_apply(F.silu(exl3_apply(h, W.gate)) * exl3_apply(h, W.up), W.down)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    ids = tok(PROMPT, add_special_tokens=False)["input_ids"]
    print("prompt:", PROMPT)
    print("token ids:", ids)
    input_ids = torch.tensor(ids, device=DEV, dtype=torch.long)
    T = input_ids.shape[0]
    positions = torch.arange(T, device=DEV)

    emb_q = load_tensor("model.language_model.embed_tokens.qweight")
    emb_s = load_tensor("model.language_model.embed_tokens.scales").to(BF16)
    emb = emb_q[input_ids].to(BF16) * emb_s[input_ids].view(-1, 1)

    gdn_w, attn_w, mlp_w = {}, {}, {}
    in_ln_w, post_ln_w = {}, {}
    for i in range(NLAYERS):
        base = f"model.language_model.layers.{i}"
        in_ln_w[i] = load_tensor(base + ".input_layernorm.weight")
        post_ln_w[i] = load_tensor(base + ".post_attention_layernorm.weight")
        if LAYER_TYPES[i] == "linear_attention":
            gdn_w[i] = GdnLayerW(i)
        else:
            attn_w[i] = AttnLayerW(i)
        mlp_w[i] = MlpW(i)
        torch.cuda.empty_cache()
        print(f"loaded layer {i} ({LAYER_TYPES[i]})", flush=True)

    final_norm_w = load_tensor("model.language_model.norm.weight")
    lm_w = load_shard("lm_head")

    print(f"running forward... (dense={DENSE}, save_int={SAVE_INT})", flush=True)
    xin = emb
    layer_states = {"embed": emb.cpu()}
    for i in range(NLAYERS):
        resid = xin
        normed = gemma_rms(resid, in_ln_w[i])
        if LAYER_TYPES[i] == "linear_attention":
            attn_out = gdn_layer(normed, gdn_w[i])
        else:
            attn_out = full_attn_layer(normed, attn_w[i], positions)
        resid = resid + attn_out
        normed2 = gemma_rms(resid, post_ln_w[i])
        mlp_out = mlp_layer(normed2, mlp_w[i])
        xout = resid + mlp_out
        if SAVE_INT:
            layer_states[i] = {"state_in": xin.cpu(), "attn_out": attn_out.cpu(),
                               "mlp_out": mlp_out.cpu(), "state_out": xout.cpu()}
        else:
            layer_states[i] = xout.cpu()
        xin = xout
        _free_layer(gdn_w.get(i))
        _free_layer(attn_w.get(i))
        _free_layer(mlp_w[i])
        if i % 8 == 0:
            print(f"  layer {i} done", flush=True)

    h = gemma_rms(xin, final_norm_w)
    layer_states["final_hidden"] = h.cpu()

    logits = exl3_apply(h, lm_w)
    last = logits[-1].float()
    logprobs = torch.log_softmax(last, dim=-1)

    topk = logprobs.topk(TOPK)
    print("\n=== ORACLE TOP-K (last token) ===")
    for lp, tid in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"  {tid:>7}  {lp:9.4f}  {repr(tok.decode([tid]))}")

    torch.save({"layer_states": layer_states,
                "prompt": PROMPT, "token_ids": ids,
                "logits_last": last.cpu()},
               os.path.join(OUT_DIR, "oracle_layer_states.pt"))
    print(f"\nsaved layer states -> {OUT_DIR}/oracle_layer_states.pt")


if __name__ == "__main__":
    main()
