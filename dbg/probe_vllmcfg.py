import sys
from vllm.transformers_utils.config import get_config
cfg = get_config("/model", trust_remote_code=True)
print("=== vllm get_config result ===")
print("top class:", type(cfg).__name__)
print("top rope_parameters:", getattr(cfg, "rope_parameters", "N/A"))
tc = cfg.get_text_config()
print("get_text_config() is top:", tc is cfg)
print("text class:", type(tc).__name__)
print("text rope_parameters:", getattr(tc, "rope_parameters", "N/A"))
print("has text_config attr:", hasattr(cfg, "text_config"))
if hasattr(cfg, "text_config") and cfg.text_config is not None:
    print("cfg.text_config rope_parameters:", getattr(cfg.text_config, "rope_parameters", "N/A"))
