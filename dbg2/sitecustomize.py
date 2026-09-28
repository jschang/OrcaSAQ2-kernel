"""Engine-side hooks for the per-layer dump run.

Env:
  LAYER_DUMP_OUT=/dbg2/server_layer_states.pt  -> wrap Qwen3_5DecoderLayer.forward,
      dump state_in / attn_out / mlp_out / state_out per layer.
  ROPE_FIX_LOG=/dbg2/rope_fix.log              -> instrument get_rope: log the
      rope_parameters it sees and the class it returns, and re-inject the mrope
      keys (from the live vLLM hf_config) when the attn config lost them.

A polling thread installs both once the model modules appear in sys.modules
(sitecustomize runs at interpreter startup, before vllm is importable).
Eager mode only (torch.compile inlines forwards).
"""
import os
import sys
import threading
import time

LAYER_OUT = os.environ.get("LAYER_DUMP_OUT", "")
ROPE_LOG = os.environ.get("ROPE_FIX_LOG", "")


def _rlog(msg):
    if not ROPE_LOG:
        return
    try:
        with open(ROPE_LOG, "a") as f:
            f.write(str(msg) + "\n")
    except Exception:
        pass


def _install_layer_dump():
    import torch
    from vllm.model_executor.models import qwen3_5 as M

    L = M.Qwen3_5DecoderLayer
    if getattr(L, "_ld_wrapped", False):
        return
    orig = L.forward
    dump = {}
    counter = [0]

    def wrapper(self, hidden_states=None, residual=None, positions=None, **kw):
        out_h, out_res = orig(
            self, hidden_states=hidden_states, residual=residual,
            positions=positions, **kw)
        try:
            state_in = (
                hidden_states if residual is None
                else residual + hidden_states)
            li = self.layer_idx
            dump[li] = {
                "state_in": state_in.detach().cpu(),
                "attn_out": (out_res - state_in).detach().cpu(),
                "mlp_out": out_h.detach().cpu(),
                "state_out": (out_res + out_h).detach().cpu(),
            }
            counter[0] += 1
            if counter[0] % 16 == 0:
                torch.save({"dumps": dump, "n": counter[0]}, LAYER_OUT)
        except Exception as e:
            print(f"[layer-dump] dump failed: {e}", flush=True)
        return out_h, out_res

    L.forward = wrapper
    L._ld_wrapped = True
    print(f"[layer-dump] wrapped Qwen3_5DecoderLayer.forward -> {LAYER_OUT}",
          flush=True)


def _install_rope_fix():
    from vllm.model_executor.models import qwen3_next as qn
    from vllm.model_executor.models import qwen3_5 as q5

    if getattr(qn.get_rope, "_rope_wrapped", False):
        return

    orig_get_rope = qn.get_rope
    calls = [0]

    def patched(*args, **kwargs):
        calls[0] += 1
        try:
            rp = kwargs.get("rope_parameters")
            if rp is None and len(args) >= 4:
                rp = args[3]
            _rlog(f"pid={os.getpid()} get_rope[{calls[0]}] "
                  f"rope_parameters={rp}")
            if rp is not None and isinstance(rp, dict) \
                    and "mrope_section" not in rp:
                try:
                    from vllm.config import get_current_vllm_config
                    top_rp = getattr(
                        get_current_vllm_config().model_config.hf_config,
                        "rope_parameters", None)
                    if isinstance(top_rp, dict):
                        for key in ("mrope_section", "mrope_interleaved"):
                            if key in top_rp and key not in rp:
                                rp[key] = top_rp[key]
                                _rlog(f"  re-injected {key} from hf_config")
                except Exception as e:
                    _rlog(f"  re-inject ERR {e!r}")
        except Exception as e:
            _rlog(f"  log ERR {e!r}")
        res = orig_get_rope(*args, **kwargs)
        _rlog(f"  -> {type(res).__name__}")
        return res

    patched._rope_wrapped = True
    qn.get_rope = patched
    if hasattr(q5, "get_rope"):
        q5.get_rope = patched

    # also record what the text config carries, once per process
    if not getattr(q5, "_rope_cfg_logged", False):
        q5._rope_cfg_logged = True
        orig_base_init = q5.Qwen3_5ForCausalLMBase.__init__

        def logged_init(self, *, vllm_config, prefix=""):
            try:
                mc = vllm_config.model_config
                _rlog(f"pid={os.getpid()} ModelConfig hf_config="
                      f"{type(mc.hf_config).__name__} "
                      f"rp={getattr(mc.hf_config, 'rope_parameters', 'N/A')}")
                _rlog(f"  hf_text_config={type(mc.hf_text_config).__name__} "
                      f"rp={getattr(mc.hf_text_config, 'rope_parameters', 'N/A')}"
                      f" same={mc.hf_text_config is mc.hf_config}")
            except Exception as e:
                _rlog(f"  cfg log ERR {e!r}")
            return orig_base_init(self, vllm_config=vllm_config, prefix=prefix)

        q5.Qwen3_5ForCausalLMBase.__init__ = logged_init

    print(f"[rope-fix] wrapped get_rope, log -> {ROPE_LOG}", flush=True)


def _loop():
    for i in range(60 * 120):  # 30 min
        if i % 40 == 0:
            _rlog(f"pid={os.getpid()} loop alive i={i} "
                  f"has_qn={'vllm.model_executor.models.qwen3_next' in sys.modules} "
                  f"has_q5={'vllm.model_executor.models.qwen3_5' in sys.modules}")
        try:
            ok = True
            if LAYER_OUT:
                if "vllm.model_executor.models.qwen3_5" in sys.modules:
                    _install_layer_dump()
                else:
                    ok = False
            if ROPE_LOG:
                if "vllm.model_executor.models.qwen3_next" in sys.modules:
                    _install_rope_fix()
                else:
                    ok = False
            if ok:
                return
        except Exception as e:
            _rlog(f"install ERR {e!r}")
        time.sleep(0.25)
    print("[hooks] gave up waiting for model modules", flush=True)


def _marker():
    try:
        with open("/dbg2/site_marker.log", "a") as f:
            f.write(f"pid={os.getpid()} LAYER_DUMP_OUT={LAYER_OUT!r} "
                    f"ROPE_FIX_LOG={ROPE_LOG!r} t={time.time():.0f}\n")
    except Exception:
        pass
    print(f"[sitecustomize] imported pid={os.getpid()} "
          f"LAYER_DUMP_OUT={LAYER_OUT!r} ROPE_FIX_LOG={ROPE_LOG!r}",
          flush=True)


_marker()
if LAYER_OUT or ROPE_LOG:
    threading.Thread(target=_loop, daemon=True).start()
