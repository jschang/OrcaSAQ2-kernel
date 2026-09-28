#pragma once
#include <cstdint>
#include <hip/hip_fp16.h>

// Tensor core register fragments (register-to-element layouts shared with the dequant path)

template <typename T, int n>
struct Vec
{
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
};

using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragC_h = Vec<half2, 2>;

// Bitfield helpers, portable (no PTX)

// bfe.u64 over the 64-bit value (hi:lo): extract `length` bits starting at `offset`
static __forceinline__ __device__ uint32_t bfe64(uint32_t lo, uint32_t hi, int offset, int length)
{
    uint64_t value = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
    uint64_t result = (value >> offset) & ((uint64_t(1) << length) - 1);
    return static_cast<uint32_t>(result);
}

// shf.r.wrap.b32 dst, lo, hi, imm : funnel right shift of the 64-bit value (hi:lo) by imm
#define FSHF_IMM(dst, lo, hi, imm) (dst) = (uint32_t)((((uint64_t)(hi)) << 32) | (uint64_t)(lo)) >> (imm)

// bfe.u32 dst, src, imm, 16
#define BFE16_IMM(dst, src, imm) (dst) = ((src) >> (imm)) & 0xffffu
