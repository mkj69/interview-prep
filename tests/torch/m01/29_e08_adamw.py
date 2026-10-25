import torch

class MyAdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = [p for p in params]
        self.lr, (self.b1, self.b2), self.eps, self.wd = lr, betas, eps, weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]   # 一阶矩（梯度的 EMA）
        self.v = [torch.zeros_like(p) for p in self.params]   # 二阶矩（梯度平方的 EMA）
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            p.mul_(1 - self.lr * self.wd)                 # 解耦的 weight decay：直接缩参数，不经过 m/v
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            m_hat = m / (1 - self.b1 ** self.t)           # 偏差修正：抵消 m, v 初始化为 0 带来的偏小
            v_hat = v / (1 - self.b2 ** self.t)
            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

torch.manual_seed(0)
w1 = torch.randn(5, 3, requires_grad=True)
w2 = w1.detach().clone().requires_grad_(True)
mine = MyAdamW([w1], lr=1e-2, weight_decay=0.1)
ref = torch.optim.AdamW([w2], lr=1e-2, weight_decay=0.1)
x = torch.randn(8, 5)
for _ in range(20):
    for w, opt in ((w1, mine), (w2, ref)):
        opt.zero_grad()
        (x @ w).pow(2).mean().backward()
        opt.step()
assert torch.allclose(w1, w2, atol=1e-6)
