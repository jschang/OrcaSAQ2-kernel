// #14 bench: MODE-tagged variants of gemv_partial_kernel for cost breakdown.
//
// MODE 0: exact copy of the production kernel (control).
// MODE 1: no dequant -- trivial ALU replaces dq_dispatch + shuffle (same smem stores, same barriers).
// MODE 2: no FMA -- dequant runs, x load + FMA removed (same barriers).
// MODE 3: x staged in smem (kills the per-tile L2 latency chain), scalar FMA as in MODE 0.
// MODE 4: MODE 3 + v_dot2_f32_f16 with column-pair thread ownership.
//
// Standalone bench extension (gemv_bench); not part of the production build.

#include <hip/hip_fp16.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "ptx.cuh"
#include "quant/reconstruct.cuh"
#include "quant/exl3_dq.cuh"
#include "quant/bits_k.cuh"
#include "quant/hadamard_inner.cuh"
#include <torch/extension.h>
#include <algorithm>

#define GEMV_THREADS 256
#define GEMV_N_TILE 128
#define GEMV_K_SPLIT 16
#define GEMV_TARGET_CTAS 512

// fp32-accumulate dot of two half2 pairs: (a.x*b.x + a.y*b.y + c); f16 products are exact in f32.
__device__ __forceinline__ float dot2_f32(uint32_t a, uint32_t b, float c)
{
    float r;
    asm("v_dot2_f32_f16 %0, %1, %2, %3\n" : "=v"(r) : "v"(a), "v"(b), "v"(c));
    return r;
}

// Dequant one 16(k) x 128(n) tile into row-major smem s_tile[16][8][8] (half2) -- identical to
// the production gemv_dequant_tile (load + barrier + dq + shuffle + store + barrier).
template <int K, int cb, bool HALF>
__device__ __forceinline__
void gemv_dequant_tile
(
    half2* __restrict__ s_tile,
    uint32_t* __restrict__ s_packed,
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

// MODE 1 stand-in for the dequant phase: same global load, same barrier structure, same number
// of smem stores, trivial ALU in place of dq_dispatch + shuffle.
template <int K, int cb, bool HALF>
__device__ __forceinline__
void gemv_dummy_dequant
(
    half2* __restrict__ s_tile,
    uint32_t* __restrict__ s_packed,
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

    if (!(lane_id & 4))
    {
        uint32_t bits = s_packed[warp_id * (packed_size / 2) + lane_id];
        half2 d = __floats2half2_rn((float) (bits & 0xf) * 0.01f, (float) ((bits >> 4) & 0xf) * 0.01f);
        int r0 = (lane_id % 4) * 2;
        int c0 = lane_id / 8;
        s_tile[r0 * 64 + warp_id * 8 + c0] = d;
        s_tile[(r0 + 1) * 64 + warp_id * 8 + c0] = d;
        s_tile[(r0 + 8) * 64 + warp_id * 8 + c0] = d;
        s_tile[(r0 + 9) * 64 + warp_id * 8 + c0] = d;
        s_tile[r0 * 64 + warp_id * 8 + c0 + 4] = d;
        s_tile[(r0 + 1) * 64 + warp_id * 8 + c0 + 4] = d;
        s_tile[(r0 + 8) * 64 + warp_id * 8 + c0 + 4] = d;
        s_tile[(r0 + 9) * 64 + warp_id * 8 + c0 + 4] = d;
    }
    __syncthreads();
}

template <int K, int cb, bool HALF, int M, int MODE>
__global__ __launch_bounds__(GEMV_THREADS)
void gemv_partial_variant
(
    const half* __restrict__ g_a_had,
    const uint16_t* __restrict__ g_packed,
    float* __restrict__ g_c_partial,
    int k_dim,
    int n_dim,
    int packed_blocks_n,
    int kt_max
)
{
    constexpr int packed_size = 16 * K + (HALF ? 8 : 0);

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int k0 = blockIdx.y * kt_max;
    int n0 = blockIdx.x * (GEMV_N_TILE / 16);

    __shared__ uint32_t s_packed[8][packed_size / 2];
    __shared__ half2 s_tile[16][8][8];
    __shared__ float s_acc[M][GEMV_N_TILE];
    // MODE 3/4: dynamic smem holds the CTA's x slice (M rows x kt_max*16 k-rows), staged once.
    extern __shared__ half s_x_flat[];  // [M][kt_max*16], sized by the host launch

    int col = t % GEMV_N_TILE;
    int khalf = t / GEMV_N_TILE;
    int n_warp = col / 16;
    int n_half2 = (col % 16) / 2;
    int n_half = (col % 16) % 2;

    for (int i = t; i < M * GEMV_N_TILE; i += GEMV_THREADS)
        ((float*) s_acc)[i] = 0.0f;
    if constexpr (MODE >= 3)
        for (int i = t; i < M * kt_max * 16; i += GEMV_THREADS)
            s_x_flat[i] = g_a_had[(int64_t) (i / (kt_max * 16)) * k_dim + k0 * 16 + i % (kt_max * 16)];
    __syncthreads();

    float acc[M], acc1[M];  // acc: column col (MODE 4: column 2cp); acc1: MODE 4's column 2cp+1
    float acc0[M], acc3[M];  // MODE 9: per-thread dual accumulators (HBM-x vs smem-x)
    #pragma unroll
    for (int m = 0; m < M; m++) { acc[m] = 0.0f; acc1[m] = 0.0f; acc0[m] = 0.0f; acc3[m] = 0.0f; }

    g_packed += ((int64_t) k0 * packed_blocks_n + n0) * packed_size;

    for (int kt = 0; kt < kt_max; kt++)
    {
        if constexpr (MODE == 1)
            gemv_dummy_dequant<K, cb, HALF>((half2*) s_tile, (uint32_t*) s_packed, g_packed, t, lane_id, warp_id);
        else
            gemv_dequant_tile<K, cb, HALF>((half2*) s_tile, (uint32_t*) s_packed, g_packed, t, lane_id, warp_id);
        g_packed += (int64_t) packed_blocks_n * packed_size;

        if constexpr (MODE == 2 || MODE == 5)
        {
            __syncthreads();  // keep the per-tile barrier structure
        }
        else if constexpr (MODE == 3)
        {
            #pragma unroll
            for (int rr = khalf * 8; rr < (khalf + 1) * 8; rr++)
            {
                half2 w2 = s_tile[rr][n_warp][n_half2];
                float w = __half2float(n_half ? __high2half(w2) : __low2half(w2));
                #pragma unroll
                for (int m = 0; m < M; m++)
                {
                    // s_x_flat is indexed in half2 units here: half (kt*16 + rr) -> half2 index (kt*16 + rr)/2.
                    half2 xm = ((half2*) s_x_flat)[m * kt_max * 8 + kt * 8 + rr / 2];
                    acc[m] += __half2float(rr % 2 ? __high2half(xm) : __low2half(xm)) * w;
                }
            }
            __syncthreads();
        }
        else if constexpr (MODE == 4)
        {
            // Column-pair ownership: threads t < 128 each own 2 adjacent columns x 8 k-rows.
            if (t < 128)
            {
                int cp = t % (GEMV_N_TILE / 2);       // column pair (columns 2cp, 2cp+1)
                int kh = t / (GEMV_N_TILE / 2);      // row half (0: rows 0-7, 1: rows 8-15)
                int nw = (2 * cp) / 16;
                int nh = ((2 * cp) % 16) / 2;
                #pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    half2 w2a = s_tile[kh * 8 + 2 * j][nw][nh];       // (w[2j][2cp],   w[2j][2cp+1])
                    half2 w2b = s_tile[kh * 8 + 2 * j + 1][nw][nh];   // (w[2j+1][2cp], w[2j+1][2cp+1])
                    uint32_t ua = *(const uint32_t*) &w2a;
                    uint32_t ub = *(const uint32_t*) &w2b;
                    uint32_t w_lo = (ua & 0xffff) | (ub << 16);        // (w[2j][2cp],   w[2j+1][2cp])
                    uint32_t w_hi = (ua >> 16) | (ub & 0xffff0000);    // (w[2j][2cp+1], w[2j+1][2cp+1])
                    #pragma unroll
                    for (int m = 0; m < M; m++)
                    {
                        // half2 units: rows (kt*16 + kh*8 + 2j, +1) -> half2 index (kt*16 + kh*8 + 2j)/2.
                        uint32_t x2 = *(const uint32_t*) (((half2*) s_x_flat) + m * kt_max * 8 + kt * 8 + kh * 4 + j);
                        acc[m] = dot2_f32(x2, w_lo, acc[m]);
                        acc1[m] = dot2_f32(x2, w_hi, acc1[m]);
                    }
                }
            }
            __syncthreads();
        }
        else if constexpr (MODE == 8)
        {
            // Probe: dump the x value each thread's FMA read for this tile (CTA (0,0), kt=0 only).
            if (blockIdx.x == 0 && blockIdx.y == 0 && kt == 0)
            {
                #pragma unroll
                for (int rr = khalf * 8; rr < (khalf + 1) * 8; rr++)
                {
                    half2 xm = ((half2*) s_x_flat)[kt * 8 + rr / 2];
                    g_c_partial[t * 16 + (rr - khalf * 8)] = __half2float(rr % 2 ? __high2half(xm) : __low2half(xm));
                }
            }
            __syncthreads();
        }
        else if constexpr (MODE == 9)
        {
            // Dual FMA in one tile: HBM x (MODE 0 style) and smem x (MODE 3 style), same thread.
            // Per-tile deltas dumped for CTA (0,0) warp 0 at flat[(kt*32+t)*8 + {0,2}].
            int k_base = (k0 + kt) * 16;
            float d0[M], d3[M];
            #pragma unroll
            for (int m = 0; m < M; m++) { d0[m] = 0.0f; d3[m] = 0.0f; }
            #pragma unroll
            for (int rr = khalf * 8; rr < (khalf + 1) * 8; rr++)
            {
                half2 w2 = s_tile[rr][n_warp][n_half2];
                float w = __half2float(n_half ? __high2half(w2) : __low2half(w2));
                #pragma unroll
                for (int m = 0; m < M; m++)
                {
                    d0[m] += __half2float(g_a_had[(int64_t) m * k_dim + k_base + rr]) * w;
                    half2 xm = ((half2*) s_x_flat)[m * kt_max * 8 + kt * 8 + rr / 2];
                    d3[m] += __half2float(rr % 2 ? __high2half(xm) : __low2half(xm)) * w;
                }
            }
            #pragma unroll
            for (int m = 0; m < M; m++) { acc0[m] += d0[m]; acc3[m] += d3[m]; }
            if (blockIdx.x == 0 && blockIdx.y == 0 && t < 32)
            {
                #pragma unroll
                for (int m = 0; m < M; m++)
                {
                    g_c_partial[(kt * 32 + t) * 8 + m * 4] = d0[m];
                    g_c_partial[(kt * 32 + t) * 8 + m * 4 + 2] = d3[m];
                }
            }
            __syncthreads();
        }
        else if constexpr (MODE == 5)
        {
            // Probe: trivial per-tile body (staged x was written at kernel start).
            __syncthreads();
        }
        else
        {
            // MODE 0: production structure (x from HBM/L2).
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
    }

    if constexpr (MODE == 4)
    {
        if (t < 128)
        {
            int cp = t % (GEMV_N_TILE / 2);
            #pragma unroll
            for (int m = 0; m < M; m++)
            {
                atomicAdd(&s_acc[m][2 * cp], acc[m]);
                atomicAdd(&s_acc[m][2 * cp + 1], acc1[m]);
            }
        }
    }
    else
    {
        #pragma unroll
        for (int m = 0; m < M; m++)
            atomicAdd(&s_acc[m][col], acc[m]);
    }
    __syncthreads();

    if constexpr (MODE == 5)
    {
        // Probe writeback: C_partial[blockIdx.y * (M*kt_max*16) + i] = staged s_x_flat[i].
        for (int i = t; i < M * kt_max * 16; i += GEMV_THREADS)
            g_c_partial[blockIdx.y * (M * kt_max * 16) + i] = __half2float(s_x_flat[i]);
    }
    else if constexpr (MODE == 8)
    {
        // Probe: the per-tile body already wrote the dump; keep the buffer untouched.
    }
    else if constexpr (MODE == 9)
    {
        // Probe: CTA (0,0) warp 0 dumps its dual full-loop accumulators (separate region).
        if (blockIdx.x == 0 && blockIdx.y == 0 && t < 32)
        {
            #pragma unroll
            for (int m = 0; m < M; m++)
            {
                g_c_partial[20480 + t * 32 + m * 16] = acc0[m];
                g_c_partial[20480 + t * 32 + m * 16 + 8] = acc3[m];
            }
        }
    }
    else if constexpr (MODE == 4)
    {
        if (t < 128)
        {
            int cp = t % (GEMV_N_TILE / 2);
            #pragma unroll
            for (int m = 0; m < M; m++)
            {
                g_c_partial[((int64_t) blockIdx.y * M + m) * n_dim + blockIdx.x * GEMV_N_TILE + 2 * cp] = s_acc[m][2 * cp];
                g_c_partial[((int64_t) blockIdx.y * M + m) * n_dim + blockIdx.x * GEMV_N_TILE + 2 * cp + 1] = s_acc[m][2 * cp + 1];
            }
        }
    }
    else
    {
        #pragma unroll
        for (int m = 0; m < M; m++)
            g_c_partial[((int64_t) blockIdx.y * M + m) * n_dim + blockIdx.x * GEMV_N_TILE + col] = s_acc[m][col];
    }
}

// Verbatim copy of the production gemv_finalize_kernel (split-K reduce + output-Hadamard
// epilogue, rocm/exllamav3_ext/rocm_gemv.cu) so the A/B harness can run the MODE 0 partial in
// the exact full production context (had + partial + finalize) without the installed ext.
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

// ---------------------------------------------------------------------------
// Host launcher (mirrors exl3_gemv's split computation; bench covers mul1 cb=2 only).
// ---------------------------------------------------------------------------

// Direct <<<>>> launch per instance: static + dynamic smem are handled by the compiler-emitted
// launcher (the hipLaunchKernel-by-pointer path mis-laid-out the dynamic base in early tests).
template <int K, bool HALF, int M, int MODE>
static void launch_variant
(
    const half* a, const uint16_t* b, float* cp,
    int k_dim, int n_dim, int pbn, int kt_max,
    dim3 grid, dim3 block, hipStream_t stream
)
{
    const size_t dyn = (size_t) (MODE >= 3 ? M * kt_max * 16 * sizeof(half) : 0);
    gemv_partial_variant<K, 2, HALF, M, MODE><<<grid, block, dyn, stream>>>(a, b, cp, k_dim, n_dim, pbn, kt_max);
    hip_check(hipGetLastError());
}

template <int Kb, int M>
static void dispatch(bool half_k, int MODE, const half* a, const uint16_t* b, float* cp,
                     int k_dim, int n_dim, int pbn, int kt_max, dim3 grid, dim3 block, hipStream_t stream)
{
    if (half_k)
    {
        switch (MODE)
        {
            case 0: launch_variant<Kb, true, M, 0>(a, b, cp, k_dim, n_dim, pbn, kt_max, grid, block, stream); break;
            case 1: launch_variant<Kb, true, M, 1>(a, b, cp, k_dim, n_dim, pbn, kt_max, grid, block, stream); break;
            case 2: launch_variant<Kb, true, M, 2>(a, b, cp, k_dim, n_dim, pbn, kt_max, grid, block, stream); break;
            case 3: launch_variant<Kb, true, M, 3>(a, b, cp, k_dim, n_dim, pbn, kt_max, grid, block, stream); break;
            case 4: launch_variant<Kb, true, M, 4>(a, b, cp, k_dim, n_dim, pbn, kt_max, grid, block, stream); break;
            case 5: launch_variant<Kb, true, M, 5>(a, b, cp, k_dim, n_dim, pbn, kt_max, grid, block, stream); break;
            case 8: launch_variant<Kb, true, M, 8>(a, b, cp, k_dim, n_dim, pbn, kt_max, grid, block, stream); break;
            case 9: launch_variant<Kb, true, M, 9>(a, b, cp, k_dim, n_dim, pbn, kt_max, grid, block, stream); break;
        }
    }
    else
    {
        switch (MODE)
        {
            case 0: launch_variant<Kb, false, M, 0>(a, b, cp, k_dim, n_dim, pbn, kt_max, grid, block, stream); break;
            case 1: launch_variant<Kb, false, M, 1>(a, b, cp, k_dim, n_dim, pbn, kt_max, grid, block, stream); break;
            case 2: launch_variant<Kb, false, M, 2>(a, b, cp, k_dim, n_dim, pbn, kt_max, grid, block, stream); break;
            case 3: launch_variant<Kb, false, M, 3>(a, b, cp, k_dim, n_dim, pbn, kt_max, grid, block, stream); break;
            case 4: launch_variant<Kb, false, M, 4>(a, b, cp, k_dim, n_dim, pbn, kt_max, grid, block, stream); break;
            case 5: launch_variant<Kb, false, M, 5>(a, b, cp, k_dim, n_dim, pbn, kt_max, grid, block, stream); break;
            case 8: launch_variant<Kb, false, M, 8>(a, b, cp, k_dim, n_dim, pbn, kt_max, grid, block, stream); break;
            case 9: launch_variant<Kb, false, M, 9>(a, b, cp, k_dim, n_dim, pbn, kt_max, grid, block, stream); break;
        }
    }
}

// A_had: (M, k) half (pre-Hadamard); B: (k/16, n/16, tile_u16) uint16; returns (splits, M, n) fp32.
at::Tensor bench_partial(const at::Tensor& A_had, const at::Tensor& B, int M, int MODE)
{
    const int tile_u16 = (int) B.size(2);
    const bool half_k = (tile_u16 % 16) != 0;
    const int Kb = tile_u16 / 16;
    TORCH_CHECK(!half_k || (tile_u16 % 16 == 8), "bench: half-integer tile must be 16K+8");
    TORCH_CHECK(M == 1 || M == 2 || M == 4 || M == 8 || M == 16, "bench: M must be 1, 2, 4, 8 or 16");

    const int size_k = (int) A_had.size(1);
    const int64_t size_n = B.size(1) * 16;
    const int n_chunks = (int) (size_n / GEMV_N_TILE);
    int splits = std::max<int64_t>(1, (GEMV_TARGET_CTAS + n_chunks - 1) / n_chunks);
    splits = std::min(splits, size_k / (16 * GEMV_K_SPLIT));
    while ((size_k / 16) % splits) ++splits;
    const int kt_max = (size_k / 16) / splits;

    at::Tensor C_partial = at::empty({splits, M, size_n}, A_had.options().dtype(at::kFloat));

    const half* a_ptr = (const half*) A_had.data_ptr();
    const uint16_t* b_ptr = (const uint16_t*) B.data_ptr();
    float* cp_ptr = C_partial.data_ptr<float>();
    int k_dim_i = size_k;
    int n_dim_i = (int) size_n;
    int pbn_i = (int) B.size(1);
    int kt_max_i = kt_max;
    hipStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    dim3 grid(n_chunks, splits), block(GEMV_THREADS);
    if (M == 1)
    {
        switch (Kb)
        {
            case 3: dispatch<3, 1>(half_k, MODE, a_ptr, b_ptr, cp_ptr, k_dim_i, n_dim_i, pbn_i, kt_max_i, grid, block, stream); break;
            case 4: dispatch<4, 1>(half_k, MODE, a_ptr, b_ptr, cp_ptr, k_dim_i, n_dim_i, pbn_i, kt_max_i, grid, block, stream); break;
            default: TORCH_CHECK(false, "bench: unsupported Kb");
        }
    }
    else if (M == 2)
    {
        switch (Kb)
        {
            case 3: dispatch<3, 2>(half_k, MODE, a_ptr, b_ptr, cp_ptr, k_dim_i, n_dim_i, pbn_i, kt_max_i, grid, block, stream); break;
            case 4: dispatch<4, 2>(half_k, MODE, a_ptr, b_ptr, cp_ptr, k_dim_i, n_dim_i, pbn_i, kt_max_i, grid, block, stream); break;
            default: TORCH_CHECK(false, "bench: unsupported Kb");
        }
    }
    else if (M == 4)
    {
        switch (Kb)
        {
            case 3: dispatch<3, 4>(half_k, MODE, a_ptr, b_ptr, cp_ptr, k_dim_i, n_dim_i, pbn_i, kt_max_i, grid, block, stream); break;
            case 4: dispatch<4, 4>(half_k, MODE, a_ptr, b_ptr, cp_ptr, k_dim_i, n_dim_i, pbn_i, kt_max_i, grid, block, stream); break;
            default: TORCH_CHECK(false, "bench: unsupported Kb");
        }
    }
    else if (M == 8)
    {
        switch (Kb)
        {
            case 3: dispatch<3, 8>(half_k, MODE, a_ptr, b_ptr, cp_ptr, k_dim_i, n_dim_i, pbn_i, kt_max_i, grid, block, stream); break;
            case 4: dispatch<4, 8>(half_k, MODE, a_ptr, b_ptr, cp_ptr, k_dim_i, n_dim_i, pbn_i, kt_max_i, grid, block, stream); break;
            default: TORCH_CHECK(false, "bench: unsupported Kb");
        }
    }
    else
    {
        switch (Kb)
        {
            case 3: dispatch<3, 16>(half_k, MODE, a_ptr, b_ptr, cp_ptr, k_dim_i, n_dim_i, pbn_i, kt_max_i, grid, block, stream); break;
            case 4: dispatch<4, 16>(half_k, MODE, a_ptr, b_ptr, cp_ptr, k_dim_i, n_dim_i, pbn_i, kt_max_i, grid, block, stream); break;
            default: TORCH_CHECK(false, "bench: unsupported Kb");
        }
    }
    return C_partial;
}

// C_partial (splits, M, N) fp32, C (M, N) fp16, svh (N/16) fp16 (or empty): the production
// finalize epilogue over the bench partial's output, with the production grid (n_chunks, M).
void bench_finalize(const at::Tensor& C_partial, at::Tensor& C, const at::Tensor& svh)
{
    const int M = (int) C.size(0);
    const int N = (int) C.size(1);
    const int splits = (int) C_partial.size(0);
    TORCH_CHECK(C_partial.dim() == 3 && C_partial.size(1) == M && C_partial.size(2) == N,
                "bench_finalize: shape mismatch");
    TORCH_CHECK(N % GEMV_N_TILE == 0, "bench_finalize: N must be divisible by ", GEMV_N_TILE);
    dim3 grid(N / GEMV_N_TILE, M), block(32);
    hipStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    const float* cp = C_partial.data_ptr<float>();
    half* c = (half*) C.data_ptr();
    if (svh.numel())
        gemv_finalize_kernel<true><<<grid, block, 0, stream>>>(cp, c, (const half*) svh.data_ptr(), N, M, splits);
    else
        gemv_finalize_kernel<false><<<grid, block, 0, stream>>>(cp, c, nullptr, N, M, splits);
    hip_check(hipGetLastError());
}

PYBIND11_MODULE(gemv_bench, m)
{
    m.def("bench_partial", &bench_partial, "variant partial kernel -> (splits, M, n) fp32");
    m.def("bench_finalize", &bench_finalize, "finalize: C_partial (splits, M, n) fp32 -> C (M, n) fp16");
}
