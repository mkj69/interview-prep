import torch

def softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z = z - z.max(dim=dim, keepdim=True).values     # 平移不变性：减去最大值，最大的指数变成 e^0 = 1
    e = z.exp()
    return e / e.sum(dim=dim, keepdim=True)

def log_softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    m = z.max(dim=dim, keepdim=True).values
    lse = m + (z - m).exp().sum(dim=dim, keepdim=True).log()   # logsumexp
    return z - lse

z = torch.tensor([[1000., 1001., 1002.], [-5., 0., 5.]])
assert torch.allclose(softmax(z), torch.softmax(z, -1))
assert torch.allclose(log_softmax(z), torch.log_softmax(z, -1))
assert torch.allclose(softmax(z, dim=0), torch.softmax(z, 0))
