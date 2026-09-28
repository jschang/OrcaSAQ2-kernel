"""Single-stream decode tok/s vs the vLLM server (streams, times first->last token)."""
import sys, time, json
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8000"
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 128
URL = f"http://127.0.0.1:{PORT}/v1/completions"

prompt = ("Write a short poem about the ocean. ")

body = json.dumps({
    "model": "orca",
    "prompt": prompt,
    "max_tokens": MAXTOK,
    "temperature": 0.0,
    "stream": True,
}).encode()

req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
t0 = time.time()
first = None
last = None
n = 0
with urllib.request.urlopen(req) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            j = json.loads(payload)
        except Exception:
            continue
        choices = j.get("choices") or []
        if choices:
            n += 1
            now = time.time()
            if first is None:
                first = now
            last = now

total = time.time() - t0
decode = (last - first) if (first is not None and last is not None) else total
print(f"prompt~{len(prompt.split())}tok  out={n}  total={total:.3f}s  first_tok={ (first-t0) if first else -1:.3f}s "
      f"decode={decode:.3f}s  decode_tok/s={ n/decode if decode>0 else 0:.2f}")
