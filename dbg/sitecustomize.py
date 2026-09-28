import sys

LOG = "/debug-logs/debug-rope.log"

def _log(*a):
    msg = " ".join(str(x) for x in a)
    try:
        with open(LOG, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass
    try:
        print(msg, file=sys.stderr)
    except Exception:
        pass

def _patch():
    try:
        import inspect
        import vllm.model_executor.layers.rotary_embedding as R
        import vllm.model_executor.models.qwen3_next as qn
        import vllm.model_executor.models.qwen3_5 as q5
        from vllm.transformers_utils import config as vcfg
    except Exception as e:
        _log("[DEBUG] failed to import:", repr(e))
        return

    orig = R.get_rope
    def patched(*args, **kwargs):
        rp = kwargs.get("rope_parameters")
        duc = kwargs.get("dual_chunk_attention_config")
        res = orig(*args, **kwargs)
        try:
            src = inspect.getsourcefile(type(res).forward_cuda)
        except Exception as e:
            src = "ERR " + repr(e)
        _log("[DEBUG] get_rope rope_parameters=", rp,
             " dual_chunk=", duc,
             "->", type(res).__name__,
             "| fwd_cuda_src=", src)
        return res
    R.get_rope = patched
    qn.get_rope = patched
    if hasattr(q5, "get_rope"):
        q5.get_rope = patched

    og = vcfg.get_hf_text_config
    def og_patched(config):
        res = og(config)
        try:
            _log("[DEBUG] get_hf_text_config in_id=", id(config),
                 " in_rp=", getattr(config, "rope_parameters", "N/A"),
                 " out_id=", id(res),
                 " out_is_in=", res is config,
                 " out_rp=", getattr(res, "rope_parameters", "N/A"))
        except Exception as e:
            _log("[DEBUG] get_hf_text_config log ERR", repr(e))
        return res
    vcfg.get_hf_text_config = og_patched
    # patch the reference imported into model.py
    import vllm.config.model as m
    if hasattr(m, "get_hf_text_config"):
        m.get_hf_text_config = og_patched
    _log("[DEBUG] patched get_rope + get_hf_text_config")

_patch()

def _patch_attn():
    try:
        import vllm.model_executor.models.qwen3_next as qn
    except Exception as e:
        _log("[DEBUG] attn import fail", repr(e)); return
    orig_init = qn.Qwen3NextAttention.__init__
    def wrapped(self, config, *args, **kwargs):
        try:
            _log("[DEBUG] Qwen3NextAttention.__init__ config_id=", id(config),
                 " config_class=", type(config).__name__,
                 " rp=", getattr(config, "rope_parameters", "N/A"))
        except Exception as e:
            _log("[DEBUG] attn init log ERR", repr(e))
        return orig_init(self, config, *args, **kwargs)
    if not getattr(qn.Qwen3NextAttention.__init__, "_orca_wrapped", False):
        wrapped._orca_wrapped = True
        qn.Qwen3NextAttention.__init__ = wrapped
    _log("[DEBUG] patched Qwen3NextAttention.__init__")

_patch_attn()

def _patch_modelbase():
    try:
        import vllm.model_executor.models.qwen3_5 as q5
    except Exception as e:
        _log("[DEBUG] q5 import fail", repr(e)); return
    cls = q5.Qwen3_5ForCausalLMBase
    orig_init = cls.__init__
    def wrapped(self, *, vllm_config, prefix=""):
        try:
            mc = vllm_config.model_config
            _log("[DEBUG] MODELBASE pid=", __import__("os").getpid(),
                 " hf_config_id=", id(mc.hf_config),
                 " hf_config_rp=", getattr(mc.hf_config, "rope_parameters", "N/A"),
                 " hf_text_id=", id(mc.hf_text_config),
                 " hf_text_rp=", getattr(mc.hf_text_config, "rope_parameters", "N/A"),
                 " same_obj=", mc.hf_config is mc.hf_text_config)
        except Exception as e:
            _log("[DEBUG] modelbase log ERR", repr(e))
        return orig_init(self, vllm_config=vllm_config, prefix=prefix)
    if not getattr(cls.__init__, "_orca_wrapped", False):
        wrapped._orca_wrapped = True
        cls.__init__ = wrapped
    _log("[DEBUG] patched Qwen3_5ForCausalLMBase.__init__")

_patch_modelbase()

def _probe_textcfg():
    try:
        import vllm.model_executor.models.qwen3_5 as q5
        import vllm.config.model as m
    except Exception as e:
        _log("[DEBUG] textcfg import fail", repr(e)); return
    orig_init = m.ModelConfig.__init__
    def wrapped(self, *a, **k):
        r = orig_init(self, *a, **k)
        try:
            hf = self.hf_config
            tc = getattr(hf, "text_config", None)
            _log("[DEBUG] MODELCFG hf_text_config_id=", id(self.hf_text_config),
                 " hf.text_config_id=", None if tc is None else id(tc),
                 " same=", self.hf_text_config is tc,
                 " hf.text_config_rp=", None if tc is None else getattr(tc, "rope_parameters", "N/A"),
                 " type_text=", type(self.hf_text_config).__name__)
        except Exception as e:
            _log("[DEBUG] modelcfg log ERR", repr(e))
        return r
    if not getattr(m.ModelConfig.__init__, "_orca_wrapped", False):
        wrapped._orca_wrapped = True
        m.ModelConfig.__init__ = wrapped
    _log("[DEBUG] patched ModelConfig.__init__ (text_config identity)")

_probe_textcfg()

def _fix_mrope():
    try:
        from vllm.transformers_utils import config as vcfg
    except Exception as e:
        _log("[FIX] vcfg import fail", repr(e)); return
    orig = vcfg.get_hf_text_config
    if getattr(orig, "_orca_fix", False):
        return
    def patched(config):
        text_config = orig(config)
        try:
            top_rp = getattr(config, "rope_parameters", None)
            sub_rp = getattr(text_config, "rope_parameters", None)
            if isinstance(top_rp, dict) and isinstance(sub_rp, dict):
                for key in ("mrope_section", "mrope_interleaved"):
                    if key in top_rp and key not in sub_rp:
                        sub_rp[key] = top_rp[key]
                        _log("[FIX] re-injected", key, "into hf_text_config")
        except Exception as e:
            _log("[FIX] re-inject ERR", repr(e))
        return text_config
    patched._orca_fix = True
    vcfg.get_hf_text_config = patched
    try:
        import vllm.config.model as m
        if hasattr(m, "get_hf_text_config"):
            m.get_hf_text_config = patched
    except Exception:
        pass
    _log("[FIX] applied mrope re-inject to get_hf_text_config")

_fix_mrope()

def _fix_attn():
    try:
        import vllm.model_executor.models.qwen3_next as qn
    except Exception as e:
        _log("[FIX] attn import fail", repr(e)); return
    orig_init = qn.Qwen3NextAttention.__init__
    if getattr(orig_init, "_orca_fix2", False):
        return
    def patched(self, config, *args, **kwargs):
        try:
            from vllm.config import get_current_vllm_config
            vcfg2 = get_current_vllm_config()
            top_rp = getattr(vcfg2.model_config.hf_config, "rope_parameters", None)
            sub_rp = getattr(config, "rope_parameters", None)
            if isinstance(top_rp, dict) and isinstance(sub_rp, dict):
                for key in ("mrope_section", "mrope_interleaved"):
                    if key in top_rp and key not in sub_rp:
                        sub_rp[key] = top_rp[key]
                        _log("[FIX2] re-injected", key, "into attn config")
        except Exception as e:
            _log("[FIX2] ERR", repr(e))
        return orig_init(self, config, *args, **kwargs)
    patched._orca_fix2 = True
    qn.Qwen3NextAttention.__init__ = patched
    _log("[FIX2] applied mrope re-inject to Qwen3NextAttention.__init__")

_fix_attn()

def _conv_diagnose(x, weight, bias, query_start_loc, has_initial_state, out, call_idx):
    import torch
    import torch.nn.functional as F
    DIM, L = x.shape
    W = weight.shape[1]
    x2d = x.transpose(0, 1).contiguous().float()  # [L, DIM]
    xs = x2d
    pad = torch.zeros(W - 1, DIM, device=x.device)
    if has_initial_state is not None and has_initial_state.numel() > 0 and bool(has_initial_state[0]):
        cs = torch.zeros(W - 1, DIM, device=x.device)  # fallback
    inp_tm = torch.cat([pad, xs], dim=0)  # [L+W-1, DIM] token-major
    w = weight.float().unsqueeze(1)  # [DIM, W] -> [DIM, 1, W] depthwise
    bf = bias.float() if bias is not None else None
    inp_cm = inp_tm.transpose(0, 1).unsqueeze(0)  # [1, DIM, L+W-1] channel-major
    ref = F.conv1d(inp_cm, w, bias=bf, groups=DIM).squeeze(0).transpose(0, 1)  # [L, DIM]
    ref = F.silu(ref)
    out2d = out.transpose(0, 1).contiguous().float()
    diff = (ref - out2d).abs()
    n_big = int((diff > 1.0).sum().item())
    his = "None"
    if has_initial_state is not None and has_initial_state.numel() > 0:
        his = f"max={bool(has_initial_state.float().max().item())} any={bool(has_initial_state.any().item())} vals={[int(v) for v in has_initial_state.flatten().tolist()[:8]]}"
    _log(f"[CONV] call={call_idx} L={L} max_abs_diff={diff.max().item():.5f} n_diff_gt1={n_big} x_stride={x.stride()} x_absmax={x.float().abs().max().item():.3f} out_absmax={out2d.abs().max().item():.3f} ref_absmax={ref.abs().max().item():.3f} has_initial_state={his}")
    if L > 1:
        per_tok = diff.amax(dim=1)
        _log(f"[CONV]   L={L} per_tok_max: {[round(v,3) for v in per_tok.tolist()[:20]]}")

def _patch_conv():
    try:
        import vllm.model_executor.layers.mamba.ops.causal_conv1d as cc
        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as g
    except Exception as e:
        _log("[CONV] import fail", repr(e)); return
    if getattr(cc, "_orca_conv_wrapped", False):
        return
    cc._orca_conv_wrapped = True
    st = {"count": 0, "big": 0, "saved": False}
    orig = g.causal_conv1d_fn  # local reference used by _forward_core
    def wrapped(x, weight, bias, conv_states, query_start_loc, cache_indices=None,
                has_initial_state=None, activation=None, metadata=None, **kw):
        L = x.shape[-1]
        out = orig(x, weight, bias, conv_states=conv_states, query_start_loc=query_start_loc,
                   cache_indices=cache_indices, has_initial_state=has_initial_state,
                   activation=activation, metadata=metadata, **kw)
        if L > 1 and not st["saved"]:
            st["saved"] = True
            try:
                import torch
                torch.save({
                    "x": x.detach().clone().contiguous().cpu(),
                    "x_stride": list(x.stride()),
                    "weight": weight.detach().clone().contiguous().cpu(),
                    "bias": (bias.detach().clone().contiguous().cpu() if bias is not None else None),
                    "out_rocm": out.detach().clone().contiguous().cpu(),
                    "L": L,
                }, "/tmp/conv_layer0.pt")
                _log(f"[CONV] saved layer0 tensors to /tmp/conv_layer0.pt (L={L})")
            except Exception as e:
                _log("[CONV] save ERR", repr(e))
        if L > 1 and st["big"] < 10:
            try:
                _conv_diagnose(x, weight, bias, query_start_loc, has_initial_state, out, st["count"])
            except Exception as e:
                _log("[CONV] diagnose ERR call", st["count"], repr(e))
            st["big"] += 1
        st["count"] += 1
        return out
    g.causal_conv1d_fn = wrapped
    cc.causal_conv1d_fn = wrapped
    _log("[CONV] patched causal_conv1d_fn (local ref in qwen_gdn_linear_attn)")

def _patch_conv_update():
    try:
        import vllm.model_executor.layers.mamba.ops.causal_conv1d as cc
    except Exception as e:
        _log("[CONVU] import fail", repr(e)); return
    if getattr(cc, "_orca_convu_wrapped", False):
        return
    cc._orca_convu_wrapped = True
    orig = cc.causal_conv1d_update
    st = {"count": 0}
    def wrapped(x, conv_state, weight, bias, activation=None, **kw):
        out = orig(x, conv_state, weight, bias, activation=activation, **kw)
        if st["count"] < 6:
            _log(f"[CONVU] call={st['count']} x_shape={tuple(x.shape)} x_stride={x.stride()} out_absmax={out.float().abs().max().item():.4f}")
        st["count"] += 1
        return out
    cc.causal_conv1d_update = wrapped
    _log("[CONVU] patched causal_conv1d_update")

def _patch_fwdcore():
    try:
        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as g
    except Exception as e:
        _log("[FWD] import fail", repr(e)); return
    cls = g.QwenGatedDeltaNetAttention
    orig = cls._forward_core
    if getattr(orig, "_orca_fwd_wrapped", False):
        return
    orig._orca_fwd_wrapped = True
    st = {"count": 0}
    def wrapped(self, mixed_qkv, b, a, core_attn_out):
        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gg
        fc = gg.get_forward_context()
        md = fc.attn_metadata
        nm = ns = 0
        if isinstance(md, dict):
            md = md.get(self.prefix)
        if md is not None:
            nm = getattr(md, "num_prefills", 0)
            ns = getattr(md, "num_decodes", 0)
        if nm > 0 and st.get("prefill_logged", 0) < 6:
            st["prefill_logged"] = st.get("prefill_logged", 0) + 1
            _log(f"[FWD-PREFILL] prefix={self.prefix} num_prefills={nm} num_decodes={ns} mixed_qkv_shape={tuple(mixed_qkv.shape)}")
        st["count"] += 1
        return orig(self, mixed_qkv, b, a, core_attn_out)
    cls._forward_core = wrapped
    _log("[FWD] patched _forward_core")

_patch_conv()
_patch_conv_update()
_patch_fwdcore()
