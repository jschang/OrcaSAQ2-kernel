"""In-process vLLM run of the probe prompt with per-layer state dumping.

Env: LAYER_DUMP_OUT / ROPE_FIX_LOG (consumed by the patched qwen3_next.py),
HIP_VISIBLE_DEVICES. Runs the exact engine the API server would use (minus
HTTP), so this output is the ROCm server's answer.

Also probes the main-process ModelConfig: ids + rope_parameters, and plants a
marker in the rope_parameters dicts so the EngineCore side can report whether
it sees the same dict object (fork) or a re-derived one (spawn).
"""
import json
import os
import torch

MODEL = os.environ.get("MODEL_PATH", "/model")
OUT = os.environ.get("LAYER_DUMP_OUT", "/dbg2/server_layer_states.pt")


def _p(msg):
    print(f"[main-probe] {msg}", flush=True)
    try:
        with open(os.environ.get("ROPE_FIX_LOG", "/dbg2/rope_fix.log"), "a") as f:
            f.write(f"[main] {msg}\n")
    except Exception:
        pass


from vllm import LLM, SamplingParams  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402

_o_cec = EngineArgs.create_engine_config


def _traced_cec(self, *args, **kwargs):
    vc = _o_cec(self, *args, **kwargs)
    mc = vc.model_config
    _p(f"ModelConfig id={id(mc)}")
    _p(f"hf_config  id={id(mc.hf_config)} class={type(mc.hf_config).__module__}.{type(mc.hf_config).__name__}")
    _p(f"hf_text    id={id(mc.hf_text_config)} class={type(mc.hf_text_config).__module__}.{type(mc.hf_text_config).__name__}")
    _p(f"hf_config  rp={getattr(mc.hf_config, 'rope_parameters', None)}")
    _p(f"hf_text    rp={getattr(mc.hf_text_config, 'rope_parameters', None)}")
    try:
        mc.hf_text_config.rope_parameters["__fork_marker__"] = "main"
        mc.hf_config.rope_parameters["__fork_marker__"] = "main"
        _p("marker planted in both rope_parameters dicts")
    except Exception as e:
        _p(f"marker ERR {e!r}")
    return vc


EngineArgs.create_engine_config = _traced_cec

import vllm.v1.engine.utils as _EU  # noqa: E402

_o_gmc = _EU.get_mp_context


def _traced_gmc():
    c = _o_gmc()
    _p(f"get_mp_context -> {getattr(c, '_actual_context', '?')}")
    return c


_EU.get_mp_context = _traced_gmc


def main():
    prompt = "The capital of France is"
    llm = LLM(
        model=MODEL,
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        max_num_seqs=1,
        seed=0,
    )
    sp = SamplingParams(max_tokens=1, temperature=0, logprobs=10)
    out = llm.generate([prompt], sp)
    o = out[0].outputs[0]
    toks = list(o.logprobs[0].items()) if o.logprobs else []
    toks = sorted(toks, key=lambda kv: kv[1].logprob, reverse=True)
    print("=== SERVER (in-process, ROCm) TOP-10 ===", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    print("token ids:", llm.get_tokenizer().encode(prompt))
    for tid, lp in toks[:10]:
        print(f"  {tid:>7}  {lp.logprob:9.4f}  {repr(tok.decode([tid]))}", flush=True)

    if os.path.exists(OUT):
        d = torch.load(OUT, map_location="cpu")
        torch.save({"dumps": d["dumps"], "n": d["n"]}, OUT)
        print(f"dump: {len(d['dumps'])} layers, n_forwards={d['n']}", flush=True)
    else:
        print("WARNING: no layer dump file produced", flush=True)


if __name__ == "__main__":
    main()
