"""Minimal `exllamav3` package shim for the ROCm build.

The real exllamav3 is a CUDA-only extension with no ROCm wheel, so this package exists only to
satisfy `from exllamav3.ext import exllamav3_ext` for the orcasaq2 plugin. It carries just the
four EXL3 GEMM functions the plugin calls (reconstruct, had_r_128, hgemm_recon, exl3_gemm),
hipified and built against this image's ROCm torch. It is not the full exllamav3 engine.
"""
