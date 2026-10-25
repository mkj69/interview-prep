import torch
import torch.nn as nn
import torch.nn.functional as F

# 1) 过拟合一个 batch：loss 应该能降到接近 0；降不下去说明管线有 bug
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 5))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
xb, yb = torch.randn(16, 20), torch.randint(0, 5, (16,))
for step in range(300):
    loss = F.cross_entropy(model(xb), yb)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"overfit loss {loss.item():.4f}")     # 远小于 0.01
assert loss.item() < 0.01

# 2) 初始 loss 检查：C 类均匀预测时 CE ≈ ln C
fresh = nn.Linear(20, 5)
nn.init.zeros_(fresh.weight); nn.init.zeros_(fresh.bias)
print(F.cross_entropy(fresh(xb), yb).item(), torch.log(torch.tensor(5.)).item())   # 1.6094 1.6094

# 3) 形状断言：让错误在第一时间爆出来
def attention_scores(q, k):
    B, H, T, D = q.shape
    assert k.shape == (B, H, T, D), f"k shape {tuple(k.shape)} != {(B, H, T, D)}"
    return q @ k.transpose(-2, -1) / D ** 0.5

# 4) 定位 NaN：anomaly 模式会在"backward 产出 NaN"的那个 op 处报错，并打印它 forward 时的调用栈（很慢，只在调试时开）
with torch.autograd.detect_anomaly():
    a = torch.tensor([0.0, 1.0], requires_grad=True)
    try:
        (a * torch.log(a)).sum().backward()   # forward: 0 * log 0 = 0 * (-inf) = NaN；backward 里 LogBackward 算 0/0
    except RuntimeError as err:
        print("anomaly:", str(err)[:60])      # Function 'LogBackward0' returned nan values in its 0th output.
# 常见 NaN 来源：lr 太大 → 发散；log(0) / sqrt(0) 的梯度 / 除以 0；fp16 溢出；全 -inf 的 softmax 行

# 5) 日志里用 .item()（或 .detach()），不要把带图的 loss 存进列表
history = []
history.append(loss.item())                  # ✓ Python float
# history.append(loss)                       # ✗ 每个 loss 都拖着整张计算图 → 显存泄漏
