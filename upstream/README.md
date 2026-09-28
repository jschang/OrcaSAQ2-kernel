# Upstream-ready patches

Two independent, verified fixes produced while bringing the OrcaSAQ-2-27B (EXL3)
checkpoint to serve correctly on **ROCm/gfx1100**. Each is packaged three ways:
a minimal apply-able patch, a PR description, and a **nested clone of the upstream
repo** with the fix committed on a dedicated branch, ready to open as a PR against
the respective upstream.

| # | Upstream | Fix | File (in upstream) | Patch | PR | Clone + branch |
|---|----------|-----|--------------------|-------|----|----------------|
| A | `turboderp-org/exllamav3` | EXL3 4-bit (k=4.0) dequant miscompile on gfx1100 | `exllamav3/exllamav3_ext/quant/exl3_dq.cuh` | [`exllamav3/0001-…patch`](exllamav3/0001-exllamav3-exl3-4bit-dequant-rocm-gfx1100.patch) | [`exllamav3/PR.md`](exllamav3/PR.md) | `exllamav3/exllamav3/` @ `orcarouter-rocm-fixes` |
| B | `vllm-project/vllm` | Qwen3.5 RoPE 2-D `positions` crash on non-CUDA platforms | `vllm/model_executor/models/qwen3_next.py` | [`vllm/0001-…patch`](vllm/0001-vllm-qwen3-next-rope-2d-positions.patch) | [`vllm/PR.md`](vllm/PR.md) | `vllm/vllm/` @ `orcarouter-rocm-fixes` |

Both are independent and can be submitted separately. Both are required to serve
the EXL3 OrcaSAQ-2-27B checkpoint end-to-end on this box (A fixes wrong weights;
B fixes the startup crash).

## Layout

Each upstream folder holds the patch + PR note, plus a **nested git clone** of the
upstream repo (shallow, at the upstream default branch) with the fix committed on
the same-named branch `orcarouter-rocm-fixes`:

```
upstream/
  README.md
  exllamav3/
    0001-exllamav3-exl3-4bit-dequant-rocm-gfx1100.patch
    PR.md
    exllamav3/              <- clone of github.com/turboderp-org/exllamav3 @ master
      ... (own .git; branch `orcarouter-rocm-fixes`)
  vllm/
    0001-vllm-qwen3-next-rope-2d-positions.patch
    PR.md
    vllm/                   <- clone of github.com/vllm-project/vllm @ main
      ... (own .git; branch `orcarouter-rocm-fixes`)
```

The nested clones are self-contained git repos (shallow, at the upstream default
branch) and are **not** part of this repo's history — they are gitignored. The
commit inside each clone is the PR content; its message credits the work to
`orcarouter/OrcaSAQ-2-27B`.

## Checkouts / branches

| Clone | Base (shallow) | Fix branch | Fix commit |
|-------|----------------|------------|------------|
| `exllamav3/exllamav3/` | `turboderp-org/exllamav3` @ `master` (v1.5.3) | `orcarouter-rocm-fixes` | `fe2b367` |
| `vllm/vllm/` | `vllm-project/vllm` @ `main` | `orcarouter-rocm-fixes` | `163c99d` |

The fix is already committed on `orcarouter-rocm-fixes` in each clone, so a PR can
be opened straight from that branch (push the branch to a fork, open the PR). The
single fix commit on each branch:

```bash
# A — exllamav3
git -C upstream/exllamav3/exllamav3 log --oneline master..orcarouter-rocm-fixes
#   0939b16  Fix EXL3 4-bit (k=4.0) dequant miscompile on ROCm/gfx1100

# B — vLLM
git -C upstream/vllm/vllm log --oneline main..orcarouter-rocm-fixes
#   0eee093  Fix Qwen3.5/Qwen3-Next RoPE crash on non-CUDA platforms (2-D positions)
```

## Applying the patch (standalone)

The patch files also work standalone against a clean upstream checkout:

```bash
# A — exllamav3: run from the exllamav3 package dir (where exllamav3_ext/ lives)
cd upstream/exllamav3/exllamav3/exllamav3
patch -p1 < ../../0001-exllamav3-exl3-4bit-dequant-rocm-gfx1100.patch

# B — vLLM: run from the vLLM checkout root
cd upstream/vllm/vllm
git apply ../0001-vllm-qwen3-next-rope-2d-positions.patch
```

Both were verified against the current upstream trees (`patch -p1 --dry-run` for A,
`git apply --check` for B).

## Verification evidence (this repo)

- **A (dequant):** `dbg2/reconstruct_raw.py` + `dbg2/diff_raw.py` — raw 4-bit
  dequant is bit-identical to the CUDA reference (`frac_wrong = 0`); full-model
  oracle top-1 flips `" the"` → `" Paris"`.
- **B (RoPE):** `dbg2/rope_shapes.txt` — the captured profile-run shapes
  (`positions=(3, 2048)`, `query=(2048, 6144)`, `key=(2048, 1024)`) that pinned the
  2-D-positions root cause.

## Provenance

- **A** is the already-merged in-repo fix: commit `b81c739`
  (`rocm/exllamav3_ext/quant/exl3_dq.cuh`). The clone branch restates it against
  the canonical exllamav3 path (`exllamav3/exllamav3_ext/quant/exl3_dq.cuh`).
- **B** was developed as an in-container monkey-patch for serving; the clone
  branch is the clean, source-level equivalent for `qwen3_next.py`'s non-fused
  `_project_qkv_gate` path.
