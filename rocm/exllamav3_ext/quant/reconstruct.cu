#include <hip/hip_fp16.h>
#include <array>
#include <cstdint>
#include "reconstruct.cuh"
// This torch is a hipify-v2 ROCm build: classes/namespaces keep their CUDA names (only the
// filenames are renamed), so the device guard and stream come from the c10::cuda / at::cuda
// namespace and masquerade over hip. c10::cuda::getCurrentCUDAStream().stream() yields the hip
// stream (cudaStream_t == hipStream_t here), which is what the <<<>>> launch below consumes.
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "bits_k.cuh"

template <int K, int cb, bool HALF = false>
__device__ __forceinline__
void reconstruct_tile
(
    half* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed,
    int packed_blocks_n,
    int packed_n_offset
)
{
    constexpr int packed_size = 16 * K + (HALF ? 8 : 0);  // in uint16s

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int k = blockIdx.y;
    int n = blockIdx.x * 8;
    int tiles_n = gridDim.x;
    int out_blocks_n = tiles_n * 8;

    // Load packed 16*128 tile
    __shared__ uint32_t s_packed[8][packed_size / 2];
    g_packed += (k * packed_blocks_n + packed_n_offset + n) * packed_size;
    if (t < packed_size)
        ((int4*) s_packed)[t] = ((int4*) g_packed)[t];
    __syncthreads();

    // Dequant
    FragB frag[2];
    dq_dispatch<K, cb, HALF>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

    // Shuffle from tensor core layout to row major tile
    __shared__ half2 tile[16][8][8];

    half2 n0 = __shfl_down_sync(0xffffffffull, frag[0][0], 4, 32);
    half2 n1 = __shfl_down_sync(0xffffffffull, frag[0][1], 4, 32);
    half2 n2 = __shfl_down_sync(0xffffffffull, frag[1][0], 4, 32);
    half2 n3 = __shfl_down_sync(0xffffffffull, frag[1][1], 4, 32);

    if (!(lane_id & 4))
    {
        half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
        half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
        half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
        half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
        half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
        half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
        half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
        half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
        int r0 = (lane_id % 4) * 2;
        int r1 = r0 + 1;
        int r2 = r0 + 8;
        int r3 = r0 + 9;
        int c0 = lane_id / 8;
        int c1 = c0 + 4;
        tile[r0][warp_id][c0] = m0;
        tile[r1][warp_id][c0] = m1;
        tile[r2][warp_id][c0] = m2;
        tile[r3][warp_id][c0] = m3;
        tile[r0][warp_id][c1] = m4;
        tile[r1][warp_id][c1] = m5;
        tile[r2][warp_id][c1] = m6;
        tile[r3][warp_id][c1] = m7;
    }
    __syncthreads();

    // Store unpacked tile
    int r = t / 16;
    int c = t % 16;
    int4* tile_int4 = (reinterpret_cast<int4*> (tile));
    int4* out_int4 = ((int4*) g_unpacked) + (k * 16 + r) * 2 * out_blocks_n + n * 2 + c;
    *out_int4 = tile_int4[t];
}

template <int K, int cb, bool HALF = false>
__global__ __launch_bounds__(256)
void reconstruct_kernel
(
    half* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed,
    int packed_blocks_n,
    int packed_n_offset
)
{
    reconstruct_tile<K, cb, HALF>(g_unpacked, g_packed, packed_blocks_n, packed_n_offset);
}

// Index cb * 8 + K - 1 for integer K; 24 + K - 1 for the half-integer rates 1.5 / 2.5 / 3.5 (mul1 only)
#define __(i, cb) reconstruct_kernel<i, cb>
constexpr auto reconstruct_kernel_instances = std::array
{
    __(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
    __(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
    __(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2),
    reconstruct_kernel<1, 2, true>, reconstruct_kernel<2, 2, true>, reconstruct_kernel<3, 2, true>
};
#undef __

/*
Reconstruct encoded+packed tensor
*/
void reconstruct_slice
(
    at::Tensor unpacked,
    at::Tensor packed,
    float K_,
    bool mcg,
    bool mul1,
    int64_t n_offset
)
{
    const at::cuda::OptionalCUDAGuard device_guard(unpacked.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // The kernel-instance tables hold 24 entries: K in 1..8 x {plain, mcg, mul1}.
    // K is derived from the checkpoint (trellis.shape[-1] // 16), so an untrusted
    // model file can drive cbi outside the table and launch whatever pointer lies
    // there. Bound both the input and the computed index.
    const BitsK bk = bits_from_K(K_);
    const int K = bk.bits;
    TORCH_CHECK(!bk.half || mul1, "reconstruct: half-integer bitrates require the mul1 codebook");
    TORCH_CHECK_SHAPES(unpacked, 0, packed, 0, 16);
    TORCH_CHECK_SIZE(packed, 2, 16 * K + (bk.half ? 8 : 0));
    TORCH_CHECK_DTYPE(unpacked, kHalf);

    int rows = packed.size(0);
    int packed_cols = packed.size(1);

    if (unpacked.numel() == 0)
        return;

    TORCH_CHECK(unpacked.size(1) % 128 == 0, "unpacked N dimension must be divisible by 128");
    TORCH_CHECK(n_offset % 128 == 0, "n_offset must be divisible by 128");
    TORCH_CHECK(n_offset >= 0, "n_offset must be non-negative");
    TORCH_CHECK(n_offset + unpacked.size(1) <= packed.size(1) * 16, "reconstruct slice exceeds packed tensor bounds");

    int cols = unpacked.size(1) / 16;
    int packed_n_offset = n_offset / 16;

    dim3 blockDim(256);
    dim3 gridDim(cols / 8, rows);

    int cbi = K - 1;
    if (bk.half) cbi += 24;
    else if (mcg) cbi += 8;
    else if (mul1) cbi += 16;
    TORCH_CHECK(cbi >= 0 && cbi < (int) reconstruct_kernel_instances.size(),
                "kernel index out of range: ", cbi);

    // Launch through a plain function pointer: clang's <<<>>> handling reliably recognizes the
    // direct name form, whereas `array[i]<<<>>>` (the nvcc idiom) it does not.
    auto kernel = reconstruct_kernel_instances[cbi];
    kernel<<<gridDim, blockDim, 0, stream>>>
    (
        (half*) unpacked.data_ptr(),
        (const uint16_t*) packed.data_ptr(),
        packed_cols,
        packed_n_offset
    );
    hip_check(hipPeekAtLastError());
}

void reconstruct
(
    at::Tensor unpacked,
    at::Tensor packed,
    float K_,
    bool mcg,
    bool mul1
)
{
    TORCH_CHECK_SHAPES(unpacked, 1, packed, 1, 16);
    reconstruct_slice(unpacked, packed, K_, mcg, mul1, 0);
}
