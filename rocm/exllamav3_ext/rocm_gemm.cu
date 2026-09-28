#include <hip/hip_fp16.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "quant/reconstruct.cuh"
#include "quant/hadamard.cuh"

// Fused decode GEMV (defined in rocm_gemv.cu): A_had @ dequant(trellis) without materializing
// the (k, n) weight in HBM. A_had must already be the input Hadamard (+ suh); C is left for the
// caller to Hadamard (+ svh).
void exl3_gemv
(
    const half* A_had,
    const at::Tensor& B,
    half* C,
    int size_m,
    int size_k,
    float K,
    bool mcg,
    bool mul1
);

/*
Row-major matmul using rocBLAS (via torch), a @ b -> c
- if c is float16, operation is float16 @ float16 -> float16 (float32 accumulate)
- if c is float32, operation is float16 @ float16 -> float32 (float32 accumulate)
*/
void hgemm_recon
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c
)
{
    const at::cuda::OptionalCUDAGuard device_guard(a.device());

    bool output_fp32 = c.dtype() == at::kFloat;
    bool output_fp16 = c.dtype() == at::kHalf;

    TORCH_CHECK(output_fp32 || output_fp16, "c must be float32 or float16");
    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(b, kHalf);
    TORCH_CHECK_DIM(b, 2);
    TORCH_CHECK(c.dim() >= 2, "c must have at least 2 dimensions");
    TORCH_CHECK_SHAPES(a, -1, b, 0, 1);
    TORCH_CHECK_SHAPES(b, 1, c, -1, 1);
    TORCH_CHECK(c.stride(-1) == 1, "c must have contiguous columns");

    int size_k = a.size(-1);
    int size_m = a.numel() / size_k;
    auto a2 = a.reshape({size_m, size_k});

    auto out = at::mm(a2, b);  // (m, n) fp16, fp32 accumulate (rocBLAS)
    c.copy_(out.to(c.scalar_type()));
    hip_check(hipPeekAtLastError());
}

/*
EXL3 matmul, A @ B -> C, unfused (Hadamard + dequant + rocBLAS).

- A: row-major A tensor, shape (m, k), dtype float16, contiguous
- B: EXL3-quantized B tensor, shape (k//16, n//16, 16*K), dtype uint16
- C: row-major C tensor, shape (m, n), dtype float16 or float32, contiguous
- suh: optional, packed input scales/flips, shape (k//16), dtype float16
- A_had: scratch for the input transform, size and dtype as A
- svh: optional, packed output scales/flips, shape (n//16), dtype float16

This is the ROCm stand-in for the fused cooperative-group kernel. It runs the exact
sequence the plugin's large-row fallback (orcasaq2/ops.py) already uses, so the result
matches the original up to the dequant + rocBLAS GEMM precision. force_shape_idx and
force_num_sms are accepted for ABI compatibility with the original signature and ignored.
*/
int exl3_gemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int force_num_sms
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());

    TORCH_CHECK_DIM(B, 3);
    TORCH_CHECK_SHAPES(A, -1, B, 0, 16);
    TORCH_CHECK_SHAPES(C, -1, B, 1, 16);
    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kShort);
    bool c_fp32 = C.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);
    TORCH_CHECK(!(mcg && mul1), "Specified both mcg and mul1");

    // K is the bitrate: 16*K uint16 per tile for integer K, 16*K + 8 for half-integer (mul1)
    const int tile_u16 = B.size(2);
    const bool half_k = (tile_u16 % 16) != 0;
    TORCH_CHECK(!half_k || (tile_u16 % 16 == 8 && mul1),
                "exl3_gemm: half-integer bitrates require the mul1 codebook");
    const float K = static_cast<float>(tile_u16 / 16) + (half_k ? 0.5f : 0.0f);

    int dim = A.dim();
    int64_t size_m = 1;
    for (int d = 0; d < dim - 1; ++d) size_m *= A.size(d);
    int size_k = A.size(-1);
    int size_n = B.size(1) * 16;

    at::Tensor A_had_t = A_had.has_value() ? A_had.value() : at::empty_like(A);
    auto A2 = A.reshape({size_m, size_k});
    auto A_had2 = A_had_t.reshape({size_m, size_k});

    // 1. input Hadamard, pre-scaled by suh, into A_had
    had_r_128(A2, A_had2, suh, c10::nullopt, 1.0f);

    // 2+3. dequant + GEMM. For small M (decode) use the fused GEMV that dequants into shared
    //      memory and dots in-kernel, so the (k, n) weight never touches HBM. Otherwise fall
    //      back to reconstruct + rocBLAS.
    const bool use_gemv = !c_fp32
        && (size_m == 1 || size_m == 2 || size_m == 4 || size_m == 8)
        && (size_n % 128 == 0)
        && (size_k % 256 == 0);
    if (use_gemv)
    {
        // void* data_ptr() avoids instantiating the unlinked data_ptr<__half>() template;
        // at::Half and HIP __half are bitwise identical.
        exl3_gemv(reinterpret_cast<const half*>(A_had2.data_ptr()), B,
                  reinterpret_cast<half*>(C.reshape({size_m, size_n}).data_ptr()),
                  (int) size_m, size_k, K, mcg, mul1);
    }
    else
    {
        at::Tensor w = at::empty({size_k, size_n}, A.options());
        reconstruct(w, B, K, mcg, mul1);
        hgemm_recon(A_had2, w, C.reshape({size_m, size_n}));
    }

    // 4. output Hadamard, post-scaled by svh, in place
    had_r_128(C.reshape({size_m, size_n}), C.reshape({size_m, size_n}), c10::nullopt, svh, 1.0f);

    return 0;
}
