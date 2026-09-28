import torch
from transformers import AutoConfig
from vllm.model_executor.layers.rotary_embedding import get_rope, MRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

CFG = "/model/config.json"
cfg = AutoConfig.from_pretrained(CFG)
print("model_type:", cfg.model_type)
rp = getattr(cfg, "rope_parameters", None)
print("config.rope_parameters =", rp)
print("has mrope_section?", rp is not None and "mrope_section" in rp)
print("max_position_embeddings:", getattr(cfg, "max_position_embeddings", None))
print("head_dim:", getattr(cfg, "head_dim", None))
dual = getattr(cfg, "dual_chunk_attention_config", None)
print("dual_chunk_attention_config =", dual)

head_size = cfg.head_dim or (cfg.hidden_size // cfg.num_attention_heads)
emb = get_rope(
    head_size=head_size,
    max_position=cfg.max_position_embeddings,
    rope_parameters=cfg.rope_parameters,
    dual_chunk_attention_config=dual,
)
print()
print("get_rope -> type:", type(emb).__name__)
print("  mro:", [c.__name__ for c in type(emb).__mro__])
print("  type is MRotaryEmbedding:", type(emb) is MRotaryEmbedding)
print("  has mrope_section attr:", getattr(emb, "mrope_section", "MISSING"))
print("  has mrope_interleaved attr:", getattr(emb, "mrope_interleaved", "MISSING"))
print("  forward_cuda src:", end=" ")
import inspect
try:
    print(inspect.getsourcefile(type(emb).forward_cuda))
except Exception as e:
    print("ERR", e)
print("  forward_hip src:", end=" ")
try:
    print(inspect.getsourcefile(type(emb).forward_hip))
except Exception as e:
    print("ERR", e)
