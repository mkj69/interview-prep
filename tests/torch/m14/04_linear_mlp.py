import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# ==== SOLUTION ====
class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        super().__init__()
        # 和 nn.Linear 一样存成 (out, in)，forward 里用 x @ W.T
        bound = 1.0 / math.sqrt(d_in)          # 让输出方差不随 d_in 增长
        self.weight = nn.Parameter(torch.empty(d_out, d_in).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(d_out).uniform_(-bound, bound)) if bias else None

    def forward(self, x):                      # x: (..., d_in) -> (..., d_out)
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias                  # 广播到最后一维
        return y

class MLP(nn.Module):
    """Transformer FFN: d -> 4d -> d, GELU。"""
    def __init__(self, d: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.fc1 = Linear(d, hidden_mult * d)
        self.fc2 = Linear(hidden_mult * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))

class SwiGLU(nn.Module):
    """LLaMA 风格 FFN：down( silu(gate(x)) * up(x) )。"""
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.gate = Linear(d, hidden, bias=False)
        self.up = Linear(d, hidden, bias=False)
        self.down = Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
# ==== TESTS ====
torch.manual_seed(0)
lin = Linear(8, 3)
x = torch.randn(2, 5, 8)
assert lin(x).shape == (2, 5, 3)
assert torch.allclose(lin(x), F.linear(x, lin.weight, lin.bias), atol=1e-6)
mlp = MLP(8)
assert sum(p.numel() for p in mlp.parameters()) == 8*32 + 32 + 32*8 + 8
names = {n for n, _ in mlp.named_parameters()}
assert names == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}   # 子模块被自动注册
out = mlp(x); out.sum().backward()
assert all(p.grad is not None for p in mlp.parameters())
sw = SwiGLU(8, 16); assert sw(x).shape == x.shape
print("ok")
