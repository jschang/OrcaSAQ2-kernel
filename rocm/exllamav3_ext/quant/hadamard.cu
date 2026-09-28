#include <hip/hip_fp16.h>
#include <cstdint>
// hipify-v2 ROCm torch: use the CUDA-masquerading guard/stream (see reconstruct.cu for why).
// ATen/Tensor.h is required here: hadamard_inner.cuh only forward-declares at::Tensor, and this
// function does member access (.size/.stride/.data_ptr) on it, which needs the full definition.
#include <ATen/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "hadamard_inner.cuh"

template <bool pre_scale, bool post_scale>
__global__ __launch_bounds__(32)
void had_hf_r_128_kernel
(
    const half* __restrict__ input_ptr,
    half* __restrict__ output_ptr,
    const half* __restrict__ scale,
    const float r_scale
)
{
    input_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    output_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_hf_r_128_inner<pre_scale, post_scale>(input_ptr, output_ptr, scale, r_scale);
}

template <bool pre_scale, bool post_scale>
__global__ __launch_bounds__(32)
void had_ff_r_128_kernel
(
    const float* __restrict__ input_ptr,
    float* __restrict__ output_ptr,
    const half* __restrict__ scale,
    const float r_scale
)
{
    input_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    output_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_ff_r_128_inner<pre_scale, post_scale>(input_ptr, output_ptr, scale, r_scale);
}

/*
Compute y = (x.view(-1, 128) @ had_128).view(x.shape)
Works inplace if y == x
x and y must be same dtype, either float16 or float32
*/
void had_r_128
(
    const at::Tensor& input,
    const at::Tensor& output,
    const c10::optional<at::Tensor>& pre_scale,
    const c10::optional<at::Tensor>& post_scale,
    const float scale
)
{
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_SHAPES_FULL(input, output);
    TORCH_CHECK_DIM(input, 2);
    TORCH_CHECK_DIV(input, 1, 128);
    int rows = input.size(0);
    int cols = input.size(1);

    int blocks = cols / 128;
    float r_scale = scale * 0.088388347648f; // scale / sqrt(128)

    dim3 blockDim(32);
    dim3 gridDim(rows, blocks);

    if (input.dtype() == at::kHalf)
    {
        TORCH_CHECK_DTYPE(output, kHalf);
        if (pre_scale.has_value())
            had_hf_r_128_kernel<true, false><<<gridDim, blockDim, 0, stream>>>
            (
                (const half*) input.data_ptr(),
                (half*) output.data_ptr(),
                (const half*) OPTPTR(pre_scale),
                r_scale
            );
        else if (post_scale.has_value())
            had_hf_r_128_kernel<false, true><<<gridDim, blockDim, 0, stream>>>
            (
                (const half*) input.data_ptr(),
                (half*) output.data_ptr(),
                (const half*) OPTPTR(post_scale),
                r_scale
            );
        else
            had_hf_r_128_kernel<false, false><<<gridDim, blockDim, 0, stream>>>
            (
                (const half*) input.data_ptr(),
                (half*) output.data_ptr(),
                (const half*) nullptr,
                r_scale
            );
        hip_check(hipPeekAtLastError());
    }

    else if (input.dtype() == at::kFloat)
    {
        TORCH_CHECK_DTYPE(output, kFloat);
        if (pre_scale.has_value())
            had_ff_r_128_kernel<true, false><<<gridDim, blockDim, 0, stream>>>
            (
                (const float*) input.data_ptr(),
                (float*) output.data_ptr(),
                (const half*) OPTPTR(pre_scale),
                r_scale
            );
        else if (post_scale.has_value())
            had_ff_r_128_kernel<false, true><<<gridDim, blockDim, 0, stream>>>
            (
                (const float*) input.data_ptr(),
                (float*) output.data_ptr(),
                (const half*) OPTPTR(post_scale),
                r_scale
            );
        else
            had_ff_r_128_kernel<false, false><<<gridDim, blockDim, 0, stream>>>
            (
                (const float*) input.data_ptr(),
                (float*) output.data_ptr(),
                (const half*) nullptr,
                r_scale
            );
        hip_check(hipPeekAtLastError());
    }

    else TORCH_CHECK(false, "unsupported datatype");
}
