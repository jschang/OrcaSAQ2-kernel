"""Dev-loop builder for the #14 GEMV bench variant kernels (dbg2/bench_gemv.cu).

Builds a standalone `gemv_bench` extension out-of-tree; does not touch the installed
exllamav3_ext.
"""
import os

import torch
from torch.utils import cpp_extension

ROOT = os.environ.get("ROCm_EXT_ROOT", "/work/rocm")
EX = os.path.join(ROOT, "exllamav3_ext")
BUILD = os.environ.get("BENCH_BUILD_DIR", "/tmp/gemvbench")

os.makedirs(BUILD, exist_ok=True)
mod = cpp_extension.load(
    name="gemv_bench",
    sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_gemv.cu")],
    extra_include_paths=[EX, os.path.join(EX, "quant")],
    extra_cuda_cflags=["-Ofast", "-DUSE_ROCM"],
    build_directory=BUILD,
    verbose=True,
)
print("LOADED:", mod.__file__)
