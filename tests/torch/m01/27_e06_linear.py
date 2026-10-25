import math
import torch
import torch.nn as nn

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))   # (out, in)，和 nn.Linear 一致
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        bound = 1 / math.sqrt(in_features)            # nn.Linear 默认初始化的上界（U(-1/√in, 1/√in)）
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):                              # x: (..., in) → (..., out)
        y = x @ self.weight.T
        return y + self.bias if self.bias is not None else y

torch.manual_seed(0)
mine, ref = MyLinear(8, 3), nn.Linear(8, 3)
ref.load_state_dict(mine.state_dict())                 # 参数名一致（weight / bias）才能直接加载
x = torch.randn(2, 5, 8)
assert torch.allclose(mine(x), ref(x), atol=1e-6)
assert len(list(mine.parameters())) == 2
assert mine.weight.abs().max() <= 1 / math.sqrt(8)
