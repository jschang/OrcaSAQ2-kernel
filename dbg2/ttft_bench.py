#!/usr/bin/env python3
"""TTFT / prefill-throughput client for the rocm-1 vLLM server.

Targets 2048-token and 8050-token prompts (MAXLEN=8192; the chat template adds
~77 tokens, so the prompt must leave room for those plus max_tokens=16).
Each measured run uses a fresh prompt variant (varying seed)
so the prefix cache cannot shortcut the prefill. Reports TTFT (time to first
non-empty streaming delta — for this reasoning model that lands in
`delta.reasoning`), total time, and per-run prompt length via /tokenize.

Usage: ttft_bench.py [base_url]
"""
import json
import random
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
API = "HandsomeLlm"
MODEL = "OrcaSAQ2-27B"

WORDS = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
         "tempor incididunt ut labore et dolore magna aliqua enim ad minim "
         "veniam quis nostrud exercitation ullamco laboris nisi aliquip ex ea "
         "commodo consequat duis aute irure in reprehenderit voluptate velit "
         "esse cillum fugiat nulla pariatur excepteur sint occaecat cupidatat "
         "non proident sunt culpa qui officia deserunt mollit anim id est laborum"
         ).split()


def post_json(path, body):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API}"})
    return json.load(urllib.request.urlopen(req))


def token_count(text):
    # this vLLM build serves the tokenizer endpoint at the root, not under /v1
    return post_json("/tokenize", {"model": MODEL, "prompt": text})["count"]


def build_prompt(target, seed):
    r = random.Random(seed)
    parts = [f"Reference seed {seed}. "]
    n = 12
    while n < target:
        L = r.choice((15, 25, 40))
        para = " ".join(r.choice(WORDS) for _ in range(L))
        parts.append(para + " ")
        n += L + 2
    text = "".join(parts)
    # converge on the exact token count (tokenizer-aware trim)
    cnt = token_count(text)
    words = text.split()
    guard = 0
    while cnt > target and guard < 40:
        drop = max(1, int(len(words) * (cnt - target) / cnt * 0.8))
        words = words[:-drop]
        text = " ".join(words)
        cnt = token_count(text)
        guard += 1
    guard = 0
    while cnt < target and guard < 40:
        text = text + " " + " ".join(r.choice(WORDS) for _ in range(20))
        cnt = token_count(text)
        guard += 1
    return text, cnt


def run_once(prompt, max_tokens=16):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True}
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API}"})
    t0 = time.time()
    first = None
    chunks = 0
    out_chars = 0
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from e
    with resp as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            d = json.loads(data)
            ch = d["choices"][0]
            delta = ch.get("delta") or {}
            if first is None and (delta.get("reasoning") or delta.get("content")
                                  or ch.get("finish_reason")):
                first = time.time()
            chunks += 1
            out_chars += len((delta.get("reasoning") or "") + (delta.get("content") or ""))
    t1 = time.time()
    return (first - t0 if first else float("nan")), t1 - t0, chunks, out_chars


def main():
    # seed offset lets re-runs avoid prefix-cache collisions with earlier invocations
    off = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    plan = [
        ("warmup", 512, 1),
        ("2048", 2048, 1 + off),
        ("2048", 2048, 2 + off),
        ("2048", 2048, 3 + off),
        ("8050", 8050, 11 + off),
        ("8050", 8050, 12 + off),
    ]
    print(f"server: {BASE}")
    print(f"{'run':>7} {'prompt':>8} {'ttft_s':>9} {'total_s':>9} {'chunks':>7} {'out_chars':>9}")
    for name, target, seed in plan:
        text, cnt = build_prompt(target, seed)
        ttft, total, chunks, out_chars = run_once(text)
        mark = "  (warmup)" if name == "warmup" else ""
        print(f"{name:>7} {cnt:>8} {ttft:>9.3f} {total:>9.3f} {chunks:>7} {out_chars:>9}{mark}")
        # short settle between runs (no server sleep; prefix cache keeps its
        # own state, unique seeds make each run a true prefill)
        time.sleep(2)


if __name__ == "__main__":
    main()
