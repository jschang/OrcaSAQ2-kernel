from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("/model")
print("after AutoConfig      :", cfg.rope_parameters)
try:
    cfg.standardize_rope_params()
    print("after standardize    :", cfg.rope_parameters)
except Exception as e:
    print("standardize EXC:", repr(e))
try:
    cfg.validate_rope()
    print("after validate       :", cfg.rope_parameters)
except Exception as e:
    print("validate EXC:", repr(e))
