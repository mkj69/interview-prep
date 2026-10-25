import torch

torch.manual_seed(0)
N, D = 500, 3
w_true = torch.tensor([2.0, -3.0, 0.5])
X = torch.randn(N, D)
y = (X @ w_true + 0.3 * torch.randn(N) > 0).float()

w = torch.zeros(D, requires_grad=True)
b = torch.zeros((), requires_grad=True)
lr = 0.5
for step in range(300):
    z = X @ w + b                                     # logits (N,)
    # BCE with logits，数值稳定写法：softplus(z) - y*z = log(1+e^z) - y z
    loss = (torch.nn.functional.softplus(z) - y * z).mean()
    loss.backward()
    with torch.no_grad():                             # 参数更新不能进计算图
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad = None                                     # 手动清零，否则梯度会累加
    b.grad = None
acc = ((X @ w + b > 0).float() == y).float().mean().item()
print(f"loss {loss.item():.3f} acc {acc:.3f} w {w.detach().numpy().round(2)}")
assert acc > 0.9
# 解析梯度对照：dL/dw = X^T (σ(z) - y) / N
z = (X @ w + b).detach()
g_manual = X.T @ (torch.sigmoid(z) - y) / N
w2 = w.detach().clone().requires_grad_(True)
(torch.nn.functional.softplus(X @ w2 + b.detach()) - y * (X @ w2 + b.detach())).mean().backward()
assert torch.allclose(w2.grad, g_manual, atol=1e-6)
