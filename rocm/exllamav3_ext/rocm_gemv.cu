// Fused decode GEMV for EXL3 on ROCm.
//
// The unfused path materializes the dequantized (k, n) weight in HBM (reconstruct) then runs a
// rocBLAS GEMM over it -- a ~2x (k,n) fp16 round trip that dominates decode. Here we dequant each
// 16x128 k-tile into shared memory (reusing the exact reconstruct_tile dequant+shuffle) and dot
// the small (m, k) input against it in-kernel, so the (k,n) weight never touches HBM. Only the
// packed trellis is read.
//
// Two kernels:
//   gemv_partial:  split-K; each CTA owns (128 cols) x (kt_max k-tiles of 16 rows), writes an
//                  fp32 partial. The split count is adaptive (see exl3_gemv): just enough CTAs
//                  to fill the device, since each split adds an HBM round trip of the partials.
//   gemv_finalize: sums the partials over the K splits and runs the output 128-point Hadamard
//                  (+ svh post-scale) in the same kernel -> fp16 C. Replaces the reduce kernel
//                  plus the separate had_r_128 launch (and the fp16 C round trip between them).
// The input Hadamard (+ suh) stays external (had_r_128), as in the unfused path.

#include <hip/hip_fp16.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "ptx.cuh"
#include "quant/reconstruct.cuh"
#include "quant/hadamard.cuh"
#include "quant/hadamard_inner.cuh"
#include "quant/exl3_dq.cuh"
#include "quant/bits_k.cuh"
#include <algorithm>

#define GEMV_THREADS 256
#define GEMV_N_TILE 128   // 8 n-tiles of 16, per CTA
#define GEMV_K_SPLIT 16   // 16-row k-tiles per CTA at the maximum (K/256) split count
// ~a full wave of the partial kernel: gfx1100 has 80 CUs x up to 8 workgroups = 640
// concurrent CTAs; targeting 512 keeps every shape's CTA count near (not far above) one wave.
#define GEMV_TARGET_CTAS 512

// Dequant one 16(k) x 128(n) tile into row-major smem s_tile[16][8][8] (half2), reusing the exact
// reconstruct_tile dequant + shuffle. g_packed points at the first of the 8 consecutive (k, n) tiles.
template <int K, int cb, bool HALF>
__device__ __forceinline__
void gemv_dequant_tile
(
    half2* __restrict__ s_tile,      // [16][8][8]
    uint32_t* __restrict__ s_packed, // [8][packed_size/2]
    const uint16_t* __restrict__ g_packed,
    int t,
    int lane_id,
    int warp_id
)
{
    constexpr int packed_size = 16 * K + (HALF ? 8 : 0);

    if (t < packed_size)
        ((int4*) s_packed)[t] = ((int4*) g_packed)[t];
    __syncthreads();

    FragB frag[2];
    dq_dispatch<K, cb, HALF>(s_packed + warp_id * (packed_size / 2), lane_id * 8, frag[0], frag[1]);

    // Shuffle from tensor core layout to row-major tile (identical to reconstruct_tile).
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
        s_tile[r0 * 64 + warp_id * 8 + c0] = m0;
        s_tile[r1 * 64 + warp_id * 8 + c0] = m1;
        s_tile[r2 * 64 + warp_id * 8 + c0] = m2;
        s_tile[r3 * 64 + warp_id * 8 + c0] = m3;
        s_tile[r0 * 64 + warp_id * 8 + c1] = m4;
        s_tile[r1 * 64 + warp_id * 8 + c1] = m5;
        s_tile[r2 * 64 + warp_id * 8 + c1] = m6;
        s_tile[r3 * 64 + warp_id * 8 + c1] = m7;
    }
    __syncthreads();
}

// CTA: GEMV_THREADS (8 warps), GEMV_N_TILE (128) cols, kt_max 16-row k-tiles.
// s_tile holds one 16x128 k-tile at a time; we dequant+dot it kt_max times, accumulating.
// 2 threads per column split the 16 k-rows (8 each); per-column results combined via smem atomic.
template <int K, int cb, bool HALF, int M>
__global__ __launch_bounds__(GEMV_THREADS)
void gemv_partial_kernel
(
    const half* __restrict__ g_a_had,     // (M, k_dim) input (already Hadamard + suh)
    const uint16_t* __restrict__ g_packed, // trellis
    float* __restrict__ g_c_partial,      // (splits, M, n_dim) fp32
    int k_dim,
    int n_dim,
    int packed_blocks_n,
    int kt_max                             // 16-row k-tiles owned by this CTA = (k_dim/16)/splits
)
{
    constexpr int packed_size = 16 * K + (HALF ? 8 : 0);

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int k0 = blockIdx.y * kt_max;                  // first 16-row k-tile index for this CTA
    int n0 = blockIdx.x * (GEMV_N_TILE / 16);      // first n-tile index (8 per CTA)

    __shared__ uint32_t s_packed[8][packed_size / 2];
    __shared__ half2 s_tile[16][8][8];
    __shared__ float s_acc[M][GEMV_N_TILE];

    int col = t % GEMV_N_TILE;
    int khalf = t / GEMV_N_TILE;
    int n_warp = col / 16;
    int n_half2 = (col % 16) / 2;
    int n_half = (col % 16) % 2;

    for (int i = t; i < M * GEMV_N_TILE; i += GEMV_THREADS)
        ((float*) s_acc)[i] = 0.0f;
    __syncthreads();

    float acc[M];
    #pragma unroll
    for (int m = 0; m < M; m++) acc[m] = 0.0f;

    g_packed += ((int64_t) k0 * packed_blocks_n + n0) * packed_size;

    for (int kt = 0; kt < kt_max; kt++)
    {
        gemv_dequant_tile<K, cb, HALF>((half2*) s_tile, (uint32_t*) s_packed, g_packed, t, lane_id, warp_id);
        g_packed += (int64_t) packed_blocks_n * packed_size;

        int k_base = (k0 + kt) * 16;
        #pragma unroll
        for (int rr = khalf * 8; rr < (khalf + 1) * 8; rr++)
        {
            half2 w2 = s_tile[rr][n_warp][n_half2];
            float w = __half2float(n_half ? __high2half(w2) : __low2half(w2));
            #pragma unroll
            for (int m = 0; m < M; m++)
                acc[m] += __half2float(g_a_had[(int64_t) m * k_dim + k_base + rr]) * w;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < M; m++)
        atomicAdd(&s_acc[m][col], acc[m]);
    __syncthreads();

    #pragma unroll
    for (int m = 0; m < M; m++)
    {
        float v = s_acc[m][col];
        g_c_partial[((int64_t) blockIdx.y * M + m) * n_dim + blockIdx.x * GEMV_N_TILE + col] = v;
    }
}

// Fused split-K reduce + output-Hadamard epilogue.
//
// C_partial (splits, M, N) fp32 -> C (M, N) fp16, one CTA per (128-col block, row): each lane
// sums its 4 owned columns (4*l .. 4*l+3) over the K splits, then runs the exact 128-point
// Hadamard of had_hf_r_128_inner (4-point in-thread + 32-point warp shuffle), scales by
// 1/sqrt(128), rounds to fp16, post-scales by svh in fp16 and stores the half4 -- the same
// order of operations as the old reduce + had_r_128 pair, minus one launch and the fp16 C
// round trip. (The fp16 C round trip also dropped one intermediate rounding, so the fused
// result is marginally closer to the exact value.)
template <bool post_scale>
__global__ __launch_bounds__(32)
void gemv_finalize_kernel
(
    const float* __restrict__ g_c_partial,
    half* __restrict__ g_c,
    const half* __restrict__ svh,
    int n_dim,
    int M,
    int splits
)
{
    int m = blockIdx.y;
    int l = threadIdx.x;
    int col0 = blockIdx.x * GEMV_N_TILE + 4 * l;

    float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
    for (int s = 0; s < splits; s++)
    {
        const float* p = g_c_partial + ((int64_t) s * M + m) * n_dim + col0;
        v0 += p[0];
        v1 += p[1];
        v2 += p[2];
        v3 += p[3];
    }

    // 4 element had (had_hf_r_128_inner)
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;

    // 32 element had, warp shuffle (had_hf_r_128_inner)
    shuffle_had_f4x32(h0, h1, h2, h3, l);
    float r_scale = 0.088388347648f; // 1.0 / sqrt(128); caller passes scale 1.0
    half2 x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
    half2 y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);

    // Post scale (had_hf_r_128_inner)
    if constexpr (post_scale)
    {
        int i = blockIdx.x * 32 + l;
        half4 scales = ((half4*) svh)[i];
        x = __hmul2(x, scales.x);
        y = __hmul2(y, scales.y);
    }

    ((half2*) (g_c + (int64_t) m * n_dim + col0))[0] = x;
    ((half2*) (g_c + (int64_t) m * n_dim + col0))[1] = y;
}

using GemvFn = void (*)(const half*, const uint16_t*, float*, int, int, int, int);
template <int K, int cb, bool HALF, int M>
static GemvFn inst() { return (GemvFn) gemv_partial_kernel<K, cb, HALF, M>; }

#define G__(k, c, m) (inst<k, c, false, m>())
static const GemvFn gemv_plain[5][8] =
{
    { G__(1, 0, 1), G__(2, 0, 1), G__(3, 0, 1), G__(4, 0, 1), G__(5, 0, 1), G__(6, 0, 1), G__(7, 0, 1), G__(8, 0, 1) },
    { G__(1, 0, 2), G__(2, 0, 2), G__(3, 0, 2), G__(4, 0, 2), G__(5, 0, 2), G__(6, 0, 2), G__(7, 0, 2), G__(8, 0, 2) },
    { G__(1, 0, 4), G__(2, 0, 4), G__(3, 0, 4), G__(4, 0, 4), G__(5, 0, 4), G__(6, 0, 4), G__(7, 0, 4), G__(8, 0, 4) },
    { G__(1, 0, 8), G__(2, 0, 8), G__(3, 0, 8), G__(4, 0, 8), G__(5, 0, 8), G__(6, 0, 8), G__(7, 0, 8), G__(8, 0, 8) },
    { G__(1, 0, 16), G__(2, 0, 16), G__(3, 0, 16), G__(4, 0, 16), G__(5, 0, 16), G__(6, 0, 16), G__(7, 0, 16), G__(8, 0, 16) },
};
static const GemvFn gemv_mcg[5][8] =
{
    { G__(1, 1, 1), G__(2, 1, 1), G__(3, 1, 1), G__(4, 1, 1), G__(5, 1, 1), G__(6, 1, 1), G__(7, 1, 1), G__(8, 1, 1) },
    { G__(1, 1, 2), G__(2, 1, 2), G__(3, 1, 2), G__(4, 1, 2), G__(5, 1, 2), G__(6, 1, 2), G__(7, 1, 2), G__(8, 1, 2) },
    { G__(1, 1, 4), G__(2, 1, 4), G__(3, 1, 4), G__(4, 1, 4), G__(5, 1, 4), G__(6, 1, 4), G__(7, 1, 4), G__(8, 1, 4) },
    { G__(1, 1, 8), G__(2, 1, 8), G__(3, 1, 8), G__(4, 1, 8), G__(5, 1, 8), G__(6, 1, 8), G__(7, 1, 8), G__(8, 1, 8) },
    { G__(1, 1, 16), G__(2, 1, 16), G__(3, 1, 16), G__(4, 1, 16), G__(5, 1, 16), G__(6, 1, 16), G__(7, 1, 16), G__(8, 1, 16) },
};
static const GemvFn gemv_mul1[5][8] =
{
    { G__(1, 2, 1), G__(2, 2, 1), G__(3, 2, 1), G__(4, 2, 1), G__(5, 2, 1), G__(6, 2, 1), G__(7, 2, 1), G__(8, 2, 1) },
    { G__(1, 2, 2), G__(2, 2, 2), G__(3, 2, 2), G__(4, 2, 2), G__(5, 2, 2), G__(6, 2, 2), G__(7, 2, 2), G__(8, 2, 2) },
    { G__(1, 2, 4), G__(2, 2, 4), G__(3, 2, 4), G__(4, 2, 4), G__(5, 2, 4), G__(6, 2, 4), G__(7, 2, 4), G__(8, 2, 4) },
    { G__(1, 2, 8), G__(2, 2, 8), G__(3, 2, 8), G__(4, 2, 8), G__(5, 2, 8), G__(6, 2, 8), G__(7, 2, 8), G__(8, 2, 8) },
    { G__(1, 2, 16), G__(2, 2, 16), G__(3, 2, 16), G__(4, 2, 16), G__(5, 2, 16), G__(6, 2, 16), G__(7, 2, 16), G__(8, 2, 16) },
};
static const GemvFn gemv_half[5][4] =
{
    { (inst<1, 2, true, 1>()), (inst<2, 2, true, 1>()), (inst<3, 2, true, 1>()), (GemvFn) nullptr },
    { (inst<1, 2, true, 2>()), (inst<2, 2, true, 2>()), (inst<3, 2, true, 2>()), (GemvFn) nullptr },
    { (inst<1, 2, true, 4>()), (inst<2, 2, true, 4>()), (inst<3, 2, true, 4>()), (GemvFn) nullptr },
    { (inst<1, 2, true, 8>()), (inst<2, 2, true, 8>()), (inst<3, 2, true, 8>()), (GemvFn) nullptr },
    { (inst<1, 2, true, 16>()), (inst<2, 2, true, 16>()), (inst<3, 2, true, 16>()), (GemvFn) nullptr },
};
#undef G__

static int m_index(int m)
{
    return m == 1 ? 0 : m == 2 ? 1 : m == 4 ? 2 : m == 8 ? 3 : 4;
}

// Fused EXL3 GEMV: C = hadamard_svh(A_had @ dequant(trellis)), replacing the
// reconstruct + GEMM pair. A_had must already be the input Hadamard (+ suh); the output
// Hadamard (+ svh) runs in the gemv_finalize epilogue, so C is complete on return.
void exl3_gemv
(
    const half* A_had,
    const at::Tensor& B,
    half* C,
    int size_m,
    int size_k,
    float K,
    bool mcg,
    bool mul1,
    const half* svh
)
{
    const auto bits = bits_from_K(K);
    const int Kb = bits.bits;
    const bool half_k = bits.half;

    const int64_t size_n = B.size(1) * 16;
    const int64_t packed_blocks_n = B.size(1);
    TORCH_CHECK(size_n % GEMV_N_TILE == 0, "gemv: N must be divisible by ", GEMV_N_TILE);
    TORCH_CHECK(size_k % (16 * GEMV_K_SPLIT) == 0, "gemv: K must be divisible by ", 16 * GEMV_K_SPLIT);
    TORCH_CHECK(size_m == 1 || size_m == 2 || size_m == 4 || size_m == 8 || size_m == 16,
               "gemv: M must be 1/2/4/8/16");

    const int mi = m_index(size_m);
    GemvFn fn;
    if (half_k)
    {
        TORCH_CHECK(!mcg && mul1, "gemv: half-integer bitrates require mul1");
        TORCH_CHECK(Kb <= 3, "gemv: half-integer bitrates are 1.5/2.5/3.5");
        fn = gemv_half[mi][Kb - 1];
    }
    else if (mcg)
        fn = gemv_mcg[mi][Kb - 1];
    else if (mul1)
        fn = gemv_mul1[mi][Kb - 1];
    else
        fn = gemv_plain[mi][Kb - 1];
    TORCH_CHECK(fn != nullptr, "gemv: no kernel instance");

    const int n_chunks = (int) (size_n / GEMV_N_TILE);

    // Adaptive split-K: the fp32 partials round-trip HBM once per split, so use no more
    // splits than needed to fill the device. The split count must divide size_k/16 (an
    // integer tile count per CTA); the walk always terminates at size_k/16.
    int splits = std::max<int64_t>(1, (GEMV_TARGET_CTAS + n_chunks - 1) / n_chunks);
    splits = std::min(splits, size_k / (16 * GEMV_K_SPLIT));
    while ((size_k / 16) % splits) ++splits;
    const int kt_max = (size_k / 16) / splits;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    auto options = at::TensorOptions().dtype(at::kFloat).device(B.device());
    at::Tensor C_partial = at::empty({splits, size_m, size_n}, options);

    // Launch via the function pointer (<<<>>> only works for named kernels).
    const half* a_ptr = A_had;
    const uint16_t* b_ptr = (const uint16_t*) B.data_ptr();
    float* cp_ptr = C_partial.data_ptr<float>();
    int k_dim_i = size_k;
    int n_dim_i = (int) size_n;
    int pbn_i = (int) packed_blocks_n;
    int kt_max_i = kt_max;
    void* args[] = { (void*)&a_ptr, (void*)&b_ptr, (void*)&cp_ptr,
                     (void*)&k_dim_i, (void*)&n_dim_i, (void*)&pbn_i, (void*)&kt_max_i };
    hipLaunchKernel(reinterpret_cast<const void*>(fn), dim3(n_chunks, splits),
                    dim3(GEMV_THREADS), args, 0, stream);
    hip_check(hipGetLastError());

    // Fused split-K reduce + output Hadamard (+ svh) epilogue -> C.
    dim3 grid_r(n_chunks, size_m);
    if (svh != nullptr)
        gemv_finalize_kernel<true><<<grid_r, dim3(32), 0, stream>>>(
            C_partial.data_ptr<float>(), C, svh, (int) size_n, size_m, splits);
    else
        gemv_finalize_kernel<false><<<grid_r, dim3(32), 0, stream>>>(
            C_partial.data_ptr<float>(), C, nullptr, (int) size_n, size_m, splits);
    hip_check(hipGetLastError());
}
