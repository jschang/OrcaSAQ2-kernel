"""Pure fp16 GEMM roofline probe (no EXL3): what is this card's practical fp16
matmul peak? Compares against the 85 TFLOP seen in the EXL3 hgemm_recon sweep."""
import torch

DEV = torch.device("cuda", 0)
props = torch.cuda.get_device_properties(0)
print(f"device: {props.name}  SMs/CUs={props.multi_processor_count}  mem={props.total_memory/1e9:.1f} GB")

for (M, K, N) in [
    (2048, 5120, 17408),   # gate_proj shape, production M
    (2048, 5120, 12288),   # q_proj shape
    (2048, 17408, 5120),   # down_proj shape
    (4096, 5120, 17408),   # bigger M, same work per row
    (2048, 2048, 2048),    # square (bandwidth-light, pure compute)
]:
    A = torch.randn(M, K, dtype=torch.float16, device=DEV)
    B = torch.randn(K, N, dtype=torch.float16, device=DEV)
    C = torch.empty(M, N, dtype=torch.float16, device=DEV)
    for _ in range(5):
        torch.matmul(A, B, out=C)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    iters = 50
    s.record()
    for _ in range(iters):
        torch.matmul(A, B, out=C)
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    flops = 2.0 * M * K * N
    print(f"M={M:>5} K={K:>5} N={N:>6}  {ms:>8.4f} ms  {flops/(ms*1e-3)/1e12:>8.2f} TFLOP")
    del A, B, C
    torch.cuda.empty_cache()
print("done")
