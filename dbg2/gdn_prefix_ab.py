"""E2E prefix-cache A/B for the GDN h0 path (ticket #12) on rocm-1.

The cache block on this stack is 1568 tokens (mamba page alignment), so the
shared prefix P must span several full blocks for a hit to occur.

Phases:
  A  cold  (zero-state prefill of P+F)        <- ground truth
  B  warm  (prefix hit -> nonzero-h0 chunk prefill of the tail of P+F)
  -- container restart (clears all caches) --
  C  cold  (zero-state prefill of P+F again)  <- reproducibility control

Greedy (temperature 0): if the h0 path is correct, B's output stream must
match A's (and C's). If the h0 path is broken, B diverges / is incoherent.

usage: python3 gdn_prefix_ab.py [phase1|phase2]
  phase1 = runs A and B, prints metrics
  phase2 = runs C after restart
Results are saved to /tmp/gdn_prefix_ab_<phase>.json
"""
import json
import random
import subprocess
import sys
import time
import urllib.request

BASE = "http://localhost:8001/v1"
KEY = "HandsomeLlm"
MODEL = "OrcaSAQ2-27B"
MAX_TOKENS = 2000


def build_P():
    """~5000-token factual dossier with embedded verifiable facts."""
    rng = random.Random(42)
    places = ["Alder Bay", "Cormorant Point", "Fenwick Harbor", "Saltmarsh Quay",
              "Tresler Cove", "Dunmore Head", "Oyster Flat", "Kettle Sound"]
    people = ["Elias Thorn", "Marta Voss", "Jonah Pike", "Petra Lindqvist",
              "Silas Crane", "Nora Bellamy", "Hugo Fenwick", "Iris Calloway",
              "Tomas Reiner", "Agnes Whitley", "Casper Doyle", "Wanda Marsh"]
    vessels = ["the Cormorant", "the Petrel", "the Grey Heron", "the Salt Hare",
               "the Bittern", "the Lark", "the Mew Gull", "the Wren"]
    n = 40
    paras = []
    for i in range(n):
        p1, p2 = rng.sample(people, 2)
        v, loc = rng.choice(vessels), rng.choice(places)
        day, month = rng.randint(1, 28), rng.choice(["January", "February", "March"])
        num = rng.randint(3, 900)
        unit = rng.choice(["barrels", "crates", "sacks", "cases"])
        paras.append(
            f"Dossier entry {i:02d}. On {month} {day}, {p1} of {loc} delivered "
            f"{num} {unit} to {p2}, who signed for the shipment at the harbor "
            f"office. {v} was the only vessel out that day, and the log shows "
            f"the deckhand reported a fog bank closing in by mid-morning. "
            f"The entry is unsigned, which the archive notes is unusual for "
            f"the period."
        )
    fixed = [
        "Facts on record. The brass clock in the lighthouse tower has run "
        "exactly 90 seconds slow since 1984 and was last wound on March 4, 1987. "
        "The harbor master's ledger records 414 barrels of cod landed in "
        "January 1987. The clock tower was last repainted in 1969. The "
        "lighthouse keeper, Marcus Webb, began service in 1981. The trawler "
        "that ran aground on March 3, 1987 was named the Cormorant. The town "
        "bell was rung eleven times at noon that day, and the harbor master "
        "said that means no one should sail for a week. The lamp fuel is "
        "delivered twice a year by barge. The keeper's logbook is kept by "
        "Priya Shah, the assistant.",
    ]
    doc = "Background dossier for the questions that follow. Read it all.\n\n"
    for p in paras + fixed:
        doc += p + "\n\n"
    doc += "Answer only from the dossier above.\n\n"
    return doc


F = (
    "Questions, answer each in one short sentence:\n"
    "1) How many years had Marcus Webb worked at the lighthouse by 1987?\n"
    "2) What was the name of the trawler that ran aground on March 3, 1987?\n"
    "3) How many barrels of cod were landed in January 1987 per the harbor "
    "master's ledger?\n"
    "4) According to the harbor master, what do the eleven bell rings mean?"
)


def chat(text, label, max_tokens=MAX_TOKENS):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + KEY,
        },
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.load(r)
    dt = time.time() - t0
    usage = j.get("usage", {})
    msg = j["choices"][0]["message"]
    stream = (msg.get("reasoning") or "") + "\n[CONTENT] " + (msg.get("content") or "")
    print(f"== {label} == elapsed {dt:.1f}s prompt_tokens={usage.get('prompt_tokens')} "
          f"completion_tokens={usage.get('completion_tokens')} "
          f"finish={j['choices'][0].get('finish_reason')}")
    return stream, usage.get("prompt_tokens"), j["choices"][0].get("finish_reason")


def hits():
    out = subprocess.run(
        ["curl", "-s", "http://localhost:8001/metrics"],
        capture_output=True, text=True, timeout=60,
    ).stdout
    for line in out.splitlines():
        if line.startswith("vllm:prefix_cache_hits_total"):
            return float(line.rsplit(" ", 1)[1])
    return -1.0


def wait_ready(seconds=900):
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            with urllib.request.urlopen(BASE + "/models", timeout=10) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise SystemExit("server did not come up in time")


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "phase1"
    P = build_P()
    full = P + F
    print(f"phase {phase}: prompt ~{len(full.split())} words\n")
    if phase == "phase1":
        h0 = hits()
        print(f"metrics hits before A: {h0}")
        sA, ptA, frA = chat(full, "A: cold (zero-state prefill)")
        hA = hits()
        print(f"metrics hits after A: {hA}")
        sB, ptB, frB = chat(full, "B: warm (prefix hit -> h0 prefill of tail)")
        hB = hits()
        print(f"metrics hits after B: {hB}")
        print(f"B gained {hB - hA:.0f} cached tokens "
              f"(prompt was {ptB} tokens; expect ~{ptB // 1568 * 1568} = full blocks)")
        with open("/tmp/gdn_prefix_ab_p1.json", "w") as f:
            json.dump({"A": sA, "B": sB, "prompt_tokens": ptA,
                       "finA": frA, "finB": frB,
                       "hits": [h0, hA, hB]}, f, indent=1)
        print("saved /tmp/gdn_prefix_ab_p1.json")
    else:
        sC, ptC, frC = chat(full, "C: cold after restart")
        with open("/tmp/gdn_prefix_ab_p2.json", "w") as f:
            json.dump({"C": sC, "prompt_tokens": ptC, "finC": frC}, f, indent=1)
        print("saved /tmp/gdn_prefix_ab_p2.json")


if __name__ == "__main__":
    main()
