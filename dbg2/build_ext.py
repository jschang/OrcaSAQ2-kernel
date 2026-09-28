"""Dev-loop builder for the ROCm EXL3 ext.

Compiles the /work/rocm sources out-of-tree with torch's ROCm-detected
CUDAExtension (hipcc) and returns the loaded module, so a new kernel can be
iterated on without touching the installed .so in dist-packages.

Usage:
  python3 build_ext.py            # build + print module path
  python3 build_ext.py --install  # also copy the .so over the installed one
"""
import os
import shutil
import sys

import torch
from torch.utils import cpp_extension

ROOT = os.environ.get("ROCm_EXT_ROOT", "/work/rocm")
BUILD = os.environ.get("EXT_BUILD_DIR", "/tmp/opext")
NAME = "exllamav3_ext"
EX = os.path.join(ROOT, "exllamav3_ext")

sources = [
    os.path.join(EX, "bindings.cpp"),
    os.path.join(EX, "rocm_gemm.cu"),
    os.path.join(EX, "quant/reconstruct.cu"),
    os.path.join(EX, "quant/hadamard.cu"),
]
for extra in ("rocm_gemv.cu",):
    p = os.path.join(EX, extra)
    if os.path.exists(p):
        sources.append(p)

os.makedirs(BUILD, exist_ok=True)
mod = cpp_extension.load(
    name=NAME,
    sources=sources,
    extra_include_paths=[EX, os.path.join(EX, "quant")],
    extra_cuda_cflags=["-Ofast", "-DUSE_ROCM"],
    extra_cflags=["-Ofast", "-DUSE_ROCM"],
    build_directory=BUILD,
    verbose=True,
)
print("LOADED:", mod.__file__)

if "--install" in sys.argv:
    import exllamav3.ext
    target_dir = os.path.dirname(os.path.abspath(exllamav3.ext.__file__))
    dst = None
    for f in os.listdir(target_dir):
        if f.startswith("exllamav3_ext.") and f.endswith(".so"):
            dst = os.path.join(target_dir, f)
    if dst is None:
        dst = os.path.join(target_dir, f"{NAME}.{torch._C._get_build_with_python_int().tag}.so")
    shutil.copyfile(mod.__file__, dst)
    print("INSTALLED ->", dst)
