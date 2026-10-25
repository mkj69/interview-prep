import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
model = nn.Linear(10, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
X, Y = torch.randn(32, 10), torch.randn(32, 1)

# 一次性用 32 个样本算梯度
model.zero_grad()
F.mse_loss(model(X), Y).backward()
g_full = model.weight.grad.clone()

# 梯度累积：4 个 micro-batch，每个 8 个样本；loss 要除以累积步数
accum = 4
model.zero_grad()
for xb, yb in zip(X.chunk(accum), Y.chunk(accum)):
    loss = F.mse_loss(model(xb), yb) / accum
    loss.backward()            # 梯度在 .grad 里累加
assert torch.allclose(model.weight.grad, g_full, atol=1e-6)
opt.step()                     # 累积完才 step 一次
opt.zero_grad()
