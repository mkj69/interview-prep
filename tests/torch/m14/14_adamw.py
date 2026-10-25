import math
import torch
# ==== SOLUTION ====
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:                                   # 懒初始化
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["t"] += 1
                t, m, v = state["t"], state["m"], state["v"]
                p.mul_(1 - lr * wd)                             # decoupled weight decay（不进 m/v）
                m.mul_(b1).add_(g, alpha=1 - b1)                # m = b1 m + (1-b1) g
                v.mul_(b2).addcmul_(g, g, value=1 - b2)         # v = b2 v + (1-b2) g^2
                m_hat = m / (1 - b1 ** t)                       # bias correction
                v_hat = v / (1 - b2 ** t)
                p.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)
# ==== TESTS ====
torch.manual_seed(0)
w1 = torch.randn(5, 3, requires_grad=True)
w2 = w1.detach().clone().requires_grad_(True)
o1 = AdamW([w1], lr=1e-2, weight_decay=0.1)
o2 = torch.optim.AdamW([w2], lr=1e-2, weight_decay=0.1)
X = torch.randn(32, 5)
for _ in range(50):
    for w, o in ((w1, o1), (w2, o2)):
        o.zero_grad()
        ((X @ w).sin() ** 2).mean().backward()
        o.step()
assert torch.allclose(w1, w2, atol=1e-5), (w1 - w2).abs().max()
print("ok")
