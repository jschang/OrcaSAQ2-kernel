from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("/model")
tc = cfg.get_text_config()
print("get_text_config() is cfg itself:", tc is cfg)
print("has text_config attr:", hasattr(cfg, "text_config"))
print("tc.rope_parameters:", getattr(tc, "rope_parameters", "N/A"))
if hasattr(cfg, "text_config") and cfg.text_config is not None:
    print("cfg.text_config is cfg:", cfg.text_config is cfg)
    print("cfg.text_config.rope_parameters:", getattr(cfg.text_config, "rope_parameters", "N/A"))
