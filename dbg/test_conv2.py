import torch, sys
tag = sys.argv[1] if len(sys.argv) > 1 else "out"
torch.manual_seed(21)
DIM = 2*16*128 + 48*128   # 10240
L = 16
WIDTH = 4
# x: channel-last => start [L, DIM] contiguous, transpose to [DIM, L]
x2d = torch.randn(L, DIM, device="cuda", dtype=torch.bfloat16)
x = x2d.transpose(0, 1)  # [DIM, L] stride (1, DIM)
weight = torch.randn(DIM, WIDTH, device="cuda", dtype=torch.bfloat16)
bias = torch.randn(DIM, device="cuda", dtype=torch.bfloat16)
num_cache = 2
conv_states = torch.zeros(num_cache, DIM, WIDTH-1, device="cuda", dtype=torch.bfloat16)
query_start_loc = torch.tensor([0, L], device="cuda", dtype=torch.int32)
cache_indices = torch.tensor([0], device="cuda", dtype=torch.int32)
has_initial_state = torch.tensor([False], device="cuda")

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
out = causal_conv1d_fn(x, weight, bias, conv_states=conv_states,
    query_start_loc=query_start_loc, cache_indices=cache_indices,
    has_initial_state=has_initial_state, activation="silu")
# out is [DIM, L]; transpose to [L, DIM]
out2d = out.transpose(0,1).contiguous()
print(tag, "out shape", tuple(out.shape), "absmax", out2d.float().abs().max().item())

# torch reference
import torch.nn.functional as F
xs = x2d.float()
# pad left with WIDTH-1 zeros
padded = F.pad(xs, (WIDTH-1, 0))  # [L, DIM, L+3]
w = weight.float().transpose(0,1)  # [WIDTH, DIM]
biasf = bias.float()
conv = F.conv1d(padded, w, bias=biasf, groups=DIM)  # [1, DIM, L]
ref = torch.nn.functional.silu(conv).squeeze(0).transpose(0,1)  # [L, DIM]
torch.save({"out": out2d.cpu(), "ref": ref.cpu()}, f"/tmp/conv2_{tag}.pt")
print(tag, "saved")
