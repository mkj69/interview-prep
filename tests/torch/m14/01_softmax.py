import torch
import torch.nn.functional as F
# ==== SOLUTION ====
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # 减去每行最大值：数学上不变，数值上 exp 的参数 <= 0，不会溢出
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp(x - m)
    return e / e.sum(dim=dim, keepdim=True)

def logsumexp(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    out = m + torch.log(torch.exp(x - m).sum(dim=dim, keepdim=True))
    return out if keepdim else out.squeeze(dim)

def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # log softmax = x - logsumexp(x)；不要写成 torch.log(softmax(x))
    return x - logsumexp(x, dim=dim, keepdim=True)
# ==== TESTS ====
torch.manual_seed(0)
x = torch.randn(4, 7) * 5
assert torch.allclose(softmax(x), F.softmax(x, dim=-1), atol=1e-6)
assert torch.allclose(softmax(x, dim=0), F.softmax(x, dim=0), atol=1e-6)
assert torch.allclose(log_softmax(x), F.log_softmax(x, dim=-1), atol=1e-5)
assert torch.allclose(logsumexp(x, dim=-1), torch.logsumexp(x, dim=-1), atol=1e-5)
big = torch.tensor([[1000.0, 1001.0, 1002.0]])      # 朴素实现会得到 nan
assert torch.isfinite(softmax(big)).all() and torch.isfinite(log_softmax(big)).all()
assert torch.allclose(softmax(big).sum(-1), torch.ones(1))
print("ok")
