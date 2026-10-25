import torch
import torch.nn as nn
import torch.nn.functional as F

# 有 bug 的版本（不要照抄）
def train_buggy(model, opt, loader, val_loader):
    for x, y in loader:
        probs = model(x).softmax(-1)
        loss = F.cross_entropy(probs, y)          # bug 1：CE 应传 logits
        loss.backward()                           # bug 2：从未 zero_grad → 梯度一直累加
        opt.step()
    acc = 0
    for x, y in val_loader:                       # bug 3：没有 model.eval()，也没有 no_grad
        acc += (model(x).argmax(-1) == y).float().mean()   # bug 4：acc 是带图的 tensor，且平均方式随最后一个 batch 大小而偏
    return acc / len(val_loader)

# 修正后
def train_fixed(model, opt, loader, val_loader):
    model.train()
    for x, y in loader:
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            correct += (model(x).argmax(-1) == y).sum().item()
            total += y.numel()
    model.train()
    return correct / total

torch.manual_seed(0)
X = torch.randn(256, 4); Y = (X[:, 0] > 0).long()
loader = [(X[i:i + 32], Y[i:i + 32]) for i in range(0, 200, 32)]
val = [(X[200:], Y[200:])]
m = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 2))
o = torch.optim.SGD(m.parameters(), lr=0.5)
for _ in range(20):
    acc = train_fixed(m, o, loader, val)
assert acc > 0.9
