import json
from transformers import AutoConfig, PretrainedConfig
p = "/model/config.json"
raw = json.load(open(p))
print("RAW config.json model_type:", raw.get("model_type"))
print("RAW rope_parameters:", raw.get("rope_parameters"))
print("--- AutoConfig ---")
try:
    cfg = AutoConfig.from_pretrained("/model")
    print("AutoConfig class:", type(cfg).__name__)
    print("AutoConfig model_type:", getattr(cfg, "model_type", None))
    print("AutoConfig rope_parameters:", getattr(cfg, "rope_parameters", "N/A"))
    for attr in ("text_config", "hf_text_config"):
        sub = getattr(cfg, attr, None)
        if sub is not None:
            print(f"cfg.{attr}.rope_parameters:", getattr(sub, "rope_parameters", "N/A"))
    # list attrs containing 'rope'
    print("attrs with 'rope':", [a for a in dir(cfg) if 'rope' in a.lower()])
except Exception as e:
    import traceback; traceback.print_exc()
