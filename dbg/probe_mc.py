import sys
try:
    from vllm.config import ModelConfig
    mc = ModelConfig(model="/model")
    print("=== ModelConfig ===")
    print("hf_config.rope_parameters:", getattr(mc.hf_config, "rope_parameters", "N/A"))
    print("hf_text_config class:", type(mc.hf_text_config).__name__)
    print("hf_text_config.rope_parameters:", getattr(mc.hf_text_config, "rope_parameters", "N/A"))
    print("hf_text_config is hf_config:", mc.hf_text_config is mc.hf_config)
except Exception as e:
    import traceback; traceback.print_exc()
