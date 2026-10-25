import torch
import torch.nn.functional as F

torch.manual_seed(0)
V = 6
W = torch.randn(4, V, requires_grad=True)
# 两个 micro-batch，有效 token 数不同（3 个和 1 个）
mbs = [(torch.randn(5, 4), torch.tensor([1, 2, 3, -100, -100])),
       (torch.randn(5, 4), torch.tensor([4, -100, -100, -100, -100]))]

def grad_of(loss_fn):
    W.grad = None
    loss_fn()
    return W.grad.clone()

# 目标：等价于"把两个 micro-batch 拼起来，对全部 4 个有效 token 求平均"
X = torch.cat([m[0] for m in mbs]); Y = torch.cat([m[1] for m in mbs])
g_target = grad_of(lambda: F.cross_entropy(X @ W, Y).backward())

# 朴素做法：每个 micro-batch 求平均再除以 2 → 第二个 batch 的那 1 个 token 权重过大
def naive():
    for x, y in mbs:
        (F.cross_entropy(x @ W, y) / len(mbs)).backward()
g_naive = grad_of(naive)

# 正确做法：先数全局有效 token 数，用 reduction="sum" 再除以它
def correct():
    n_tok = sum((y != -100).sum() for _, y in mbs)
    for x, y in mbs:
        (F.cross_entropy(x @ W, y, reduction="sum") / n_tok).backward()
g_ok = grad_of(correct)

assert torch.allclose(g_ok, g_target, atol=1e-6)
assert not torch.allclose(g_naive, g_target, atol=1e-4)
