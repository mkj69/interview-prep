import torch, torch.nn as nn, torch.nn.functional as F
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))     # gain only, no bias, no mean-centering

    def forward(self, x):
        xf = x.float()                                # do the reduction in fp32 even under bf16
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d, hidden=None, multiple_of=256):
        super().__init__()
        hidden = hidden or int(8 * d / 3)             # 8/3 d keeps params ~= a 4d two-matrix FFN
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)  # round up for hardware
        self.w1 = nn.Linear(d, hidden, bias=False)    # gate projection
        self.w3 = nn.Linear(d, hidden, bias=False)    # up projection
        self.w2 = nn.Linear(hidden, d, bias=False)    # down projection

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

torch.manual_seed(0)
n = RMSNorm(32); x = torch.randn(4, 5, 32) * 9
y = n(x)
assert torch.allclose(y.pow(2).mean(-1), torch.ones(4, 5), atol=1e-3)
yb = n(x.bfloat16()); assert yb.dtype == torch.bfloat16
f = SwiGLU(96)            # 8/3*96 = 256 -> hidden 256
assert f.w1.out_features == 256
assert f(x.new_zeros(2, 96)).abs().max() == 0
f2 = SwiGLU(4096)         # 10922 -> rounded to 11008 (LLaMA-7B)
assert f2.w1.out_features == 11008
print("04 rmsnorm/swiglu ok")
