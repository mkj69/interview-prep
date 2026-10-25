import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 数据：两个高斯团的二分类 ----
N = 2000
X = torch.cat([torch.randn(N // 2, 2) + 2, torch.randn(N // 2, 2) - 2])
y = torch.cat([torch.ones(N // 2), torch.zeros(N // 2)]).long()
perm = torch.randperm(N)
X, y = X[perm], y[perm]
train_dl = DataLoader(TensorDataset(X[:1600], y[:1600]), batch_size=64, shuffle=True)
val_dl = DataLoader(TensorDataset(X[1600:], y[1600:]), batch_size=256)

# ---- 模型 / 优化器 / 调度器 ----
model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 2)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
epochs = 5
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_dl))

@torch.no_grad()                      # 评估：不建图
def evaluate():
    model.eval()                      # 关闭 dropout
    correct = total = 0
    for xb, yb in val_dl:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(-1) == yb).sum().item()
        total += yb.numel()
    model.train()                     # 别忘了切回来
    return correct / total

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)                                   # 1. forward
        loss = F.cross_entropy(logits, yb)                   # 2. loss（传 logits）
        opt.zero_grad(set_to_none=True)                      # 3. 清梯度
        loss.backward()                                      # 4. backward
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 5. 梯度裁剪
        opt.step()                                           # 6. 更新参数
        sched.step()                                         # 7. 更新学习率（每个 step）
    print(f"epoch {epoch}  loss {loss.item():.4f}  val_acc {evaluate():.3f}")
assert evaluate() > 0.95
