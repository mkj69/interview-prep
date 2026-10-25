import torch
# ==== SOLUTION ====
def grpo_advantages(rewards, group_size, eps=1e-6, scale_std=True):
    """rewards: (N*G,)，同一 prompt 的 G 个样本相邻。返回 (N*G,) 的组内标准化 advantage。"""
    r = rewards.view(-1, group_size)                           # (N, G)
    adv = r - r.mean(dim=1, keepdim=True)
    if scale_std:                                              # Dr. GRPO 建议去掉这一步
        adv = adv / (r.std(dim=1, keepdim=True) + eps)
    return adv.view(-1)

def grpo_loss(logp_new, logp_old, logp_ref, advantages, mask, clip_eps=0.2, beta=0.04):
    """
    logp_*: (B, T) token log prob；advantages: (B,) 序列级，广播到每个 token。
    KL 用 k3 估计（无偏且非负）: KL ≈ exp(ref - new) - (ref - new) - 1
    按 DeepSeekMath 原式：先对每条序列的 token 取平均，再对序列取平均。
    """
    A = advantages.unsqueeze(1)                                # (B, 1)
    ratio = torch.exp(logp_new - logp_old)
    surr = torch.min(ratio * A, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * A)
    log_r = logp_ref - logp_new
    kl = torch.exp(log_r) - log_r - 1                          # k3, 每个 token >= 0
    per_tok = -(surr - beta * kl)                              # (B, T)
    m = mask.float()
    per_seq = (per_tok * m).sum(1) / m.sum(1).clamp(min=1)     # 序列内平均
    return per_seq.mean(), ((kl * m).sum() / m.sum()).detach()
# ==== TESTS ====
torch.manual_seed(0)
r = torch.tensor([1., 0., 0., 1., 1., 1., 1., 1.])            # 两组, G=4
adv = grpo_advantages(r, 4)
assert torch.allclose(adv[:4].mean(), torch.tensor(0.), atol=1e-6)
assert torch.allclose(adv[4:], torch.zeros(4))                 # 全对的组 advantage = 0，没有学习信号
# 组 1: mean=0.5, 无偏 std=0.577 -> adv = ±0.5/0.577 = ±0.866
assert torch.allclose(adv[:4], torch.tensor([0.866, -0.866, -0.866, 0.866]), atol=1e-3)
B, T = 8, 5
lp = torch.randn(B, T) - 1
mask = torch.ones(B, T); mask[0, 3:] = 0
# new == old == ref：KL = 0，loss = -mean(A)
loss, kl = grpo_loss(lp, lp, lp, adv, mask)
assert abs(kl.item()) < 1e-7 and abs(loss.item() + adv.mean().item()) < 1e-6
# k3 非负且期望等于真实 KL：用一个离散分布验证
p = torch.softmax(torch.randn(6), 0); q = torch.softmax(torch.randn(6), 0)   # p=pi_new, q=ref
true_kl = (p * (p.log() - q.log())).sum()
k3 = (p * ((q / p) - (q.log() - p.log()) - 1)).sum()           # E_{x~p}[k3]
assert torch.allclose(true_kl, k3, atol=1e-6)
print("ok")
