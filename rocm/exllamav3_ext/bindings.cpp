#include <torch/extension.h>

// Declared in the .cu translation units (reconstruct.cu, hadamard.cu, rocm_gemm.cu)
void reconstruct(at::Tensor unpacked, at::Tensor packed, float K, bool mcg, bool mul1);
void had_r_128
(
    const at::Tensor& input,
    const at::Tensor& output,
    const c10::optional<at::Tensor>& pre_scale,
    const c10::optional<at::Tensor>& post_scale,
    const float scale
);
void hgemm_recon(at::Tensor a, at::Tensor b, at::Tensor c);
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
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "exllamav3 EXL3 quant GEMM subset, hipified for ROCm";
    m.def("reconstruct", &reconstruct, "Reconstruct (dequantize) an EXL3 packed tensor");
    m.def("had_r_128", &had_r_128, "In-place/out 128-point Hadamard over the last dimension");
    m.def("hgemm_recon", &hgemm_recon, "Row-major fp16 matmul a @ b -> c via rocBLAS");
    m.def("exl3_gemm", &exl3_gemm, "EXL3 A @ B -> C (unfused Hadamard + dequant + GEMM)");
}
