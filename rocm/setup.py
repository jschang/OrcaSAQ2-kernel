# Build the hipified EXL3 GEMM subset as `exllamav3.ext.exllamav3_ext` against this image's
# ROCm torch. The dotted name is load-bearing: setuptools places the compiled .so at
# exllamav3/ext/exllamav3_ext.<abi>.so, so `from exllamav3.ext import exllamav3_ext` (what
# orcasaq2/ops.py does) resolves to it with no re-export shim.
#
# There is no torch.utils.hip_extension in this torch build; torch.utils.cpp_extension.
# CUDAExtension detects the ROCm torch (IS_HIP_EXTENSION) and drives hipcc from ROCM_HOME,
# linking amdhip64/c10_hip/torch_hip automatically. The "nvcc" key in extra_compile_args is
# the slot cpp_extension passes to the GPU compiler (hipcc here).

import os

import torch
from setuptools import setup
from torch.utils import cpp_extension

# Build only for the serving GPU. The image default PYTORCH_ROCM_ARCH lists eight arches, which
# is correct for a redistributable wheel but multiplies this build time ~8x for a single-GPU
# image; the Dockerfile pins this to the one arch we deploy on.
archs = os.environ.get("PYTORCH_ROCM_ARCH", "gfx1100")

sources = [
    "exllamav3_ext/bindings.cpp",
    "exllamav3_ext/rocm_gemm.cu",
    "exllamav3_ext/rocm_gemv.cu",
    "exllamav3_ext/quant/reconstruct.cu",
    "exllamav3_ext/quant/hadamard.cu",
]

# -DUSE_ROCM routes compat.cuh's tanh down the portable branch instead of the CUDA
# tanh.approx.f32 asm; -Ofast matches the upstream build's host/codegen settings.
extra_cflags = ["-Ofast", "-DUSE_ROCM"]
extra_gpu_cflags = ["-Ofast", "-DUSE_ROCM"]

setup(
    name="orcasaq2-exl3-rocm",
    version="1.5.1+rocm",
    description="Hipified EXL3 (exllamav3) GEMM subset for orcasaq2 on ROCm",
    packages=["exllamav3", "exllamav3.ext"],
    ext_modules=[
        cpp_extension.CUDAExtension(
            "exllamav3.ext.exllamav3_ext",
            sources,
            extra_compile_args={"cxx": extra_cflags, "nvcc": extra_gpu_cflags},
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
