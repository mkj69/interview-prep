import torch
import torch.nn as nn
# ==== SOLUTION ====
def dropout(x: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:
    if not training or p == 0.0:
        return x                                   # 推理时恒等，不需要再缩放
    if p == 1.0:
        return torch.zeros_like(x)
    keep = (torch.rand_like(x) > p).to(x.dtype)    # 以概率 1-p 保留
    return x * keep / (1.0 - p)                    # inverted dropout：训练时放大

class Dropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
    def forward(self, x):
        return dropout(x, self.p, self.training)  # self.training 由 .train()/.eval() 设置
# ==== TESTS ====
torch.manual_seed(0)
x = torch.ones(200_000)
y = dropout(x, 0.3)
assert abs((y == 0).float().mean().item() - 0.3) < 0.01      # 约 30% 被置零
assert abs(y.mean().item() - 1.0) < 0.01                    # 期望保持不变
assert torch.allclose(torch.unique(y), torch.tensor([0.0, 1 / 0.7]))     # 留下的元素被放大到 1/(1-p)
d = Dropout(0.5); d.eval()
assert torch.equal(d(x), x)
assert torch.equal(dropout(x, 1.0), torch.zeros_like(x))
print("ok")
