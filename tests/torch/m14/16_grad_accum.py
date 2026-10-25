import torch
import torch.nn as nn
import torch.nn.functional as F
# ==== SOLUTION ====
def clip_grad_norm_(params, max_norm: float, eps: float = 1e-6):
    """全局 L2 范数裁剪（所有参数拼成一个向量算范数）。返回裁剪前的范数。"""
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.sqrt(sum((g.detach().float() ** 2).sum() for g in grads))
    scale = max_norm / (total + eps)
    if scale < 1:
        for g in grads:
            g.mul_(scale)
    return total

def train_step_accum(model, opt, micro_batches, max_norm=1.0):
    """
    micro_batches: list of (x, y)。等价于在它们拼起来的大 batch 上做一步
    （前提：每个 micro-batch 大小相同，loss 为 mean）。
    """
    k = len(micro_batches)
    opt.zero_grad(set_to_none=True)
    total_loss = 0.0
    for x, y in micro_batches:
        loss = F.mse_loss(model(x), y)
        (loss / k).backward()             # 除以 k：梯度累加后等于大 batch 的平均梯度
        total_loss += loss.item() / k
    norm = clip_grad_norm_([p for p in model.parameters()], max_norm)   # 累加完、step 前裁剪
    opt.step()
    return total_loss, norm
# ==== TESTS ====
torch.manual_seed(0)
X, Y = torch.randn(32, 4), torch.randn(32, 2)
m1 = nn.Linear(4, 2); m2 = nn.Linear(4, 2); m2.load_state_dict(m1.state_dict())
o1 = torch.optim.SGD(m1.parameters(), lr=0.1); o2 = torch.optim.SGD(m2.parameters(), lr=0.1)
l1, n1 = train_step_accum(m1, o1, [(X[i:i+8], Y[i:i+8]) for i in range(0, 32, 8)], max_norm=0.5)
o2.zero_grad(); F.mse_loss(m2(X), Y).backward()
n2 = torch.nn.utils.clip_grad_norm_(m2.parameters(), 0.5); o2.step()
assert abs(n1.item() - n2.item()) < 1e-5
for p1, p2 in zip(m1.parameters(), m2.parameters()):
    assert torch.allclose(p1, p2, atol=1e-6)
print("ok")
