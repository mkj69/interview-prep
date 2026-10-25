import math, torch, torch.nn as nn
torch.manual_seed(0)
lin = nn.Linear(1024, 1024)
nn.init.kaiming_normal_(lin.weight, mode='fan_in', nonlinearity='relu')
assert abs(lin.weight.std().item() - math.sqrt(2 / 1024)) < 2e-3
# GPT-2 style residual projection scaling
n_layer = 12
proj = nn.Linear(768, 768)
nn.init.normal_(proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
assert abs(proj.weight.std().item() - 0.02 / math.sqrt(24)) < 1e-3
# clipping
model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 1))
loss = (model(torch.randn(32, 10)) * 1e4).pow(2).mean(); loss.backward()
pre = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
post = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters()))
assert pre > 1.0 and abs(post.item() - 1.0) < 1e-4
print("ok")
