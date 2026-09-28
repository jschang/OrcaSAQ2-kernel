import torch
srv = torch.load("/dbg2/server_layer_states.pt", map_location="cpu")["dumps"]
orc = torch.load("/dbg2/oracle_layer_states.pt", map_location="cpu")["layer_states"]

def stats(a, b):
    a = a.float(); b = b.float()
    if a.shape != b.shape:
        return None
    d = (a - b).abs()
    av = a[-1].flatten(); bv = b[-1].flatten()
    cos = torch.nn.functional.cosine_similarity(av, bv, dim=0).item()
    return d.max().item(), d.mean().item(), cos

KEYS = ["state_in", "attn_out", "mlp_out", "state_out"]
print(f"{'layer':>5} | {'state_in':>26} | {'attn_out':>26} | {'mlp_out':>26} | {'state_out':>26}")
for i in range(64):
    s = srv[i]; o = orc[i]
    def fmt(key):
        r = stats(s[key], o[key])
        if r is None:
            return f"{'SHAPE':>26}"
        return f"{r[0]:>9.3f} {r[1]:>9.5f} {r[2]:>8.5f}"
    print(f"{i:>5} | {fmt('state_in')} | {fmt('attn_out')} | {fmt('mlp_out')} | {fmt('state_out')}")

print("\nlayer0 state_in vs oracle embed:", stats(srv[0]["state_in"], orc["embed"]))
print("final: server state_out[63] vs oracle final_hidden:", stats(srv[63]["state_out"], orc["final_hidden"]))
