import math, torch
import torch.nn.functional as F

def lm_loss(logits, input_ids, loss_mask, z_coef=1e-4):
    lg = logits[:, :-1].float()
    tgt = input_ids[:, 1:]
    m = loss_mask[:, 1:].float()
    nll = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
                          reduction='none').view_as(m)
    logZ = torch.logsumexp(lg, dim=-1)
    ce = (nll * m).sum() / m.sum()
    z = (logZ.pow(2) * m).sum() / m.sum()
    return ce + z_coef * z, ce.detach(), z.detach()

torch.manual_seed(0)
B, T, V = 2, 6, 11
logits = torch.randn(B, T, V, requires_grad=True)
ids = torch.randint(0, V, (B, T))
mask = torch.ones(B, T, dtype=torch.long); mask[0, :3] = 0
tot, ce, z = lm_loss(logits, ids, mask)
s, n = 0.0, 0
for b in range(B):
    for t in range(T - 1):
        if mask[b, t + 1]:
            p = torch.softmax(logits[b, t].detach(), -1)
            s += -math.log(p[ids[b, t + 1]].item()); n += 1
assert abs(s / n - ce.item()) < 1e-5
tot.backward()
assert logits.grad[:, -1].abs().sum() == 0      # 最后一个位置没有目标
print("ok")
