import torch, torch.nn as nn, torch.nn.functional as F
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight.float()).to(x.dtype)
class SwiGLU(nn.Module):
    def __init__(self, d, hidden=None, multiple_of=256):
        super().__init__()
        hidden = hidden or int(8 * d / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
m = SwiGLU(4096); assert m.w_up.out_features == 11008
x = torch.randn(2, 7, 64); n = RMSNorm(64)
out = n(x); assert torch.allclose(out.pow(2).mean(-1), torch.ones(2, 7), atol=1e-3)
assert SwiGLU(64)(x).shape == x.shape
print("ok")
