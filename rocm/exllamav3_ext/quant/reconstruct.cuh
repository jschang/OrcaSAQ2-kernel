#pragma once

#include <ATen/Tensor.h>

void reconstruct
(
    at::Tensor unpacked,
    at::Tensor packed,
    float K,
    bool mcg,
    bool mul1
);

void reconstruct_slice
(
    at::Tensor unpacked,
    at::Tensor packed,
    float K,
    bool mcg,
    bool mul1,
    int64_t n_offset
);
