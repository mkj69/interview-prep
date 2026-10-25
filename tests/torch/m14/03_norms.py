import torch
import torch.nn as nn
import torch.nn.functional as F
# ==== SOLUTION ====
class LayerNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))   # gamma
        self.bias = nn.Parameter(torch.zeros(d))    # beta

    def forward(self, x):                           # x: (..., d)
        mu = x.mean(dim=-1, keepdim=True)
        var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)   # 有偏方差 (除以 d)
        x_hat = (x - mu) / torch.sqrt(var + self.eps)
        return x_hat * self.weight + self.bias

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))   # 没有 bias，不减均值

    def forward(self, x):
        dtype = x.dtype
        x = x.float()                               # 归约在 fp32 里做
        rms_inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms_inv).to(dtype) * self.weight
# ==== TESTS ====
torch.manual_seed(0)
x = torch.randn(3, 5, 16) * 4 + 2
ln = LayerNorm(16)
with torch.no_grad():
    ln.weight.copy_(torch.randn(16)); ln.bias.copy_(torch.randn(16))
ref = F.layer_norm(x, (16,), ln.weight, ln.bias, eps=1e-5)
assert torch.allclose(ln(x), ref, atol=1e-5)
rms = RMSNorm(16)
ref_rms = x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6)
assert torch.allclose(rms(x), ref_rms, atol=1e-5)
y = LayerNorm(16)(x)
assert torch.allclose(y.mean(-1), torch.zeros(3, 5), atol=1e-5)
assert torch.allclose(y.var(-1, unbiased=False), torch.ones(3, 5), atol=1e-3)
assert len(list(ln.parameters())) == 2 and len(list(rms.parameters())) == 1
print("ok")
