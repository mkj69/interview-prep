import torch, torch.nn as nn
torch.manual_seed(0)
d = nn.Dropout(p=0.3); x = torch.ones(1000, 1000)
d.train(); y = d(x)
assert abs((y == 0).float().mean().item() - 0.3) < 0.01 and abs(y.mean().item() - 1) < 0.01
assert torch.allclose(y[y != 0], torch.full_like(y[y != 0], 1 / 0.7))
d.eval(); assert torch.equal(d(x), x)
bn = nn.BatchNorm1d(4); bn.train()
for _ in range(200): bn(torch.randn(64, 4) * 2 + 3)
assert torch.allclose(bn.running_mean, torch.full((4,), 3.), atol=0.3)
bn.eval(); out = bn(torch.randn(1, 4) * 2 + 3); assert out.shape == (1, 4)
print("ok")
