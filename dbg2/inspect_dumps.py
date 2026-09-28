import torch, os
for name in ("server_layer_states.pt", "oracle_layer_states.pt"):
    p = f"/dbg2/{name}"
    print(f"\n########## {name} ##########")
    if not os.path.exists(p):
        print("  (missing)"); continue
    d = torch.load(p, map_location="cpu")
    print(" top-level keys:", list(d.keys()))
    for k in d:
        v = d[k]
        if isinstance(v, dict):
            keys = list(v.keys())
            print(f"  {k}: dict with {len(keys)} entries; sample keys: {keys[:6]} ... {keys[-3:]}")
            # inspect one entry
            sk = keys[0] if keys else None
            if sk is not None and not isinstance(sk, str) or True:
                ev = v[sk]
                if isinstance(ev, (tuple, list)):
                    print(f"     [{sk}] -> tuple/list len={len(ev)}; shapes:", [tuple(t.shape) if torch.is_tensor(t) else type(t) for t in ev])
                elif torch.is_tensor(ev):
                    print(f"     [{sk}] -> tensor shape={tuple(ev.shape)} dtype={ev.dtype}")
                else:
                    print(f"     [{sk}] -> {type(ev)}")
        elif torch.is_tensor(v):
            print(f"  {k}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)} = {v if not isinstance(v,(list,)) else f'list len {len(v)}'}")
