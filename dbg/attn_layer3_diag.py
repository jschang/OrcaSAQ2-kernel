import json
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors import safe_open

from exllamav3.ext import exllamav3_ext as ext
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.rocm_attn import (
    RocmAttentionImpl,
    RocmAttentionMetadata,
)


MODEL_PATH = os.environ.get("MODEL_PATH", "/model")
LAYER = int(os.environ.get("LAYER", "3"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "8"))


def load_index():
    with open(os.path.join(MODEL_PATH, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


IDX = load_index()
with open(os.path.join(MODEL_PATH, "config.json")) as f:
    CFG = json.load(f)

DEV = torch.device("cuda", 0)
HIDDEN = CFG["hidden_size"]
NUM_HEADS = CFG["num_attention_heads"]
NUM_KV_HEADS = CFG["num_key_value_heads"]
HEAD_DIM = CFG["head_dim"]
SCALE = HEAD_DIM ** -0.5
ROPE_PARAMS = CFG["rope_parameters"]
ROTARY_DIM = int(HEAD_DIM * ROPE_PARAMS["partial_rotary_factor"])
ROPE_BASE = float(ROPE_PARAMS["rope_theta"])


def load_tensor(key, device=DEV):
    with safe_open(os.path.join(MODEL_PATH, IDX[key]), framework="pt", device="cpu") as f:
        tensor = f.get_tensor(key)
    return tensor.to(device)


def load_shard(base):
    trellis = load_tensor(base + ".trellis").contiguous()
    suh = load_tensor(base + ".suh").to(torch.float16).contiguous()
    svh = load_tensor(base + ".svh").to(torch.float16).contiguous()
    k = trellis.shape[-1] / 16
    k = int(k) if float(k).is_integer() else k
    return {
        "trellis": trellis,
        "suh": suh,
        "svh": svh,
        "K": k,
        "mcg": base + ".mcg" in IDX,
        "mul1": base + ".mul1" in IDX,
    }


def exl3_apply(x, shard):
    rows = x.numel() // x.shape[-1]
    xf = x.reshape(rows, x.shape[-1]).half().contiguous()
    y = torch.empty((rows, shard["svh"].numel()), dtype=torch.float16, device=DEV)
    ext.exl3_gemm(
        xf,
        shard["trellis"],
        y,
        shard["suh"],
        torch.empty_like(xf),
        shard["svh"],
        -1,
        shard["mcg"],
        shard["mul1"],
        0,
    )
    return y.to(x.dtype).reshape(*x.shape[:-1], shard["svh"].numel())


def rms_head(x, weight, eps):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = y * (weight.float() + 1.0)
    return y.to(x.dtype)


def apply_rope_1d(q_flat, k_flat, positions):
    inv_freq = 1.0 / (
        ROPE_BASE
        ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32, device=DEV) / ROTARY_DIM)
    )
    freqs = torch.einsum("i,j->ij", positions.float(), inv_freq)
    cos = freqs.cos().to(q_flat.dtype)
    sin = freqs.sin().to(q_flat.dtype)

    def rotate(x_flat, heads):
        x = x_flat.view(SEQ_LEN, heads, HEAD_DIM)
        x_rot = x[..., :ROTARY_DIM]
        x_pass = x[..., ROTARY_DIM:]
        x1, x2 = torch.chunk(x_rot, 2, dim=-1)
        c = cos.unsqueeze(1)
        s = sin.unsqueeze(1)
        y_rot = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)
        return torch.cat((y_rot, x_pass), dim=-1).reshape(SEQ_LEN, heads * HEAD_DIM)

    return rotate(q_flat, NUM_HEADS), rotate(k_flat, NUM_KV_HEADS)


def summarize(name, actual, ref):
    d = (actual.detach().cpu().float() - ref.detach().cpu().float()).abs()
    print(
        name,
        "max_abs",
        float(d.max()),
        "mean_abs",
        float(d.mean()),
        "bad_gt_1e-2",
        int((d > 1e-2).sum()),
        "numel",
        d.numel(),
    )


def main():
    torch.manual_seed(321)
    prefix = f"model.language_model.layers.{LAYER}.self_attn"
    q_shard = load_shard(prefix + ".q_proj")
    k_shard = load_shard(prefix + ".k_proj")
    v_shard = load_shard(prefix + ".v_proj")
    o_shard = load_shard(prefix + ".o_proj")
    q_norm_weight = load_tensor(prefix + ".q_norm.weight")
    k_norm_weight = load_tensor(prefix + ".k_norm.weight")

    hidden_states = (
        torch.randn(SEQ_LEN, HIDDEN, device=DEV, dtype=torch.bfloat16) * 0.2
    ).contiguous()
    positions = torch.arange(SEQ_LEN, device=DEV, dtype=torch.long)

    q_gate = exl3_apply(hidden_states, q_shard)
    k = exl3_apply(hidden_states, k_shard)
    v = exl3_apply(hidden_states, v_shard)

    q, gate = torch.chunk(q_gate.view(SEQ_LEN, NUM_HEADS, 2 * HEAD_DIM), 2, dim=-1)
    q = rms_head(q, q_norm_weight, CFG["rms_norm_eps"]).reshape(
        SEQ_LEN, NUM_HEADS * HEAD_DIM
    )
    k = rms_head(k.view(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM), k_norm_weight, CFG["rms_norm_eps"])
    k = k.reshape(SEQ_LEN, NUM_KV_HEADS * HEAD_DIM)
    q_rope, k_rope = apply_rope_1d(q, k, positions)
    q3 = q_rope.view(SEQ_LEN, NUM_HEADS, HEAD_DIM).contiguous()
    k3 = k_rope.view(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM).contiguous()
    v3 = v.view(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM).contiguous()

    block_size = 16
    block_id = 1
    kv_cache = torch.zeros(
        (2, 2, block_size, NUM_KV_HEADS * HEAD_DIM), device=DEV, dtype=torch.bfloat16
    )
    metadata = RocmAttentionMetadata(
        num_actual_tokens=SEQ_LEN,
        max_query_len=SEQ_LEN,
        query_start_loc=torch.tensor([0, SEQ_LEN], device=DEV, dtype=torch.int32),
        max_seq_len=SEQ_LEN,
        seq_lens=torch.tensor([SEQ_LEN], device=DEV, dtype=torch.int32),
        block_table=torch.tensor([[block_id]], device=DEV, dtype=torch.int32),
        slot_mapping=torch.arange(SEQ_LEN, device=DEV, dtype=torch.int64)
        + block_id * block_size,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        causal=True,
    )
    layer = SimpleNamespace(
        _k_scale=torch.tensor(1.0, device=DEV, dtype=torch.float32),
        _v_scale=torch.tensor(1.0, device=DEV, dtype=torch.float32),
    )
    impl = RocmAttentionImpl(
        NUM_HEADS,
        HEAD_DIM,
        SCALE,
        NUM_KV_HEADS,
        None,
        None,
        "bfloat16",
        attn_type=AttentionType.DECODER,
    )
    attn_backend = torch.empty(
        (SEQ_LEN, NUM_HEADS, HEAD_DIM), device=DEV, dtype=torch.bfloat16
    )
    impl.forward(layer, q3, k3, v3, kv_cache, metadata, attn_backend)
    torch.cuda.synchronize()

    q_cpu = q3.detach().cpu()
    k_cpu = k3.detach().cpu()
    v_cpu = v3.detach().cpu()
    q_ref = q_cpu.permute(1, 0, 2).unsqueeze(0).float()
    k_ref = (
        k_cpu.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
        .permute(1, 0, 2)
        .unsqueeze(0)
        .float()
    )
    v_ref = (
        v_cpu.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
        .permute(1, 0, 2)
        .unsqueeze(0)
        .float()
    )
    attn_ref = F.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, dropout_p=0.0, is_causal=True, scale=SCALE
    )
    attn_ref = attn_ref.squeeze(0).permute(1, 0, 2).to(torch.bfloat16).to(DEV)

    gate_sig = torch.sigmoid(gate.reshape(SEQ_LEN, NUM_HEADS * HEAD_DIM))
    core_backend = (attn_backend.reshape(SEQ_LEN, NUM_HEADS * HEAD_DIM) * gate_sig).contiguous()
    core_ref = (attn_ref.reshape(SEQ_LEN, NUM_HEADS * HEAD_DIM) * gate_sig).contiguous()
    out_backend = exl3_apply(core_backend, o_shard)
    out_ref = exl3_apply(core_ref, o_shard)
    torch.cuda.synchronize()

    print("layer", LAYER, "seq_len", SEQ_LEN, "rotary_dim", ROTARY_DIM)
    print("qkv shapes", tuple(q_gate.shape), tuple(k.shape), tuple(v.shape))
    summarize("attn", attn_backend, attn_ref)
    summarize("gated_core", core_backend, core_ref)
    summarize("o_proj_out", out_backend, out_ref)
    print("out sample backend", out_backend.detach().cpu()[0, :8].tolist())
    print("out sample ref    ", out_ref.detach().cpu()[0, :8].tolist())


if __name__ == "__main__":
    main()
