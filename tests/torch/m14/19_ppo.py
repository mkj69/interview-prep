import torch
# ==== SOLUTION ====
def ppo_loss(logp_new, logp_old, advantages, values, returns, mask,
             entropy=None, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, values_old=None, vf_clip=None):
    """
    全部 (B, T) 的 token 级张量；mask: 1 = response token。
    logp_old / advantages / returns 来自 rollout，必须是常数（不带梯度）。
    """
    m = mask.float()
    mmean = lambda x: (x * m).sum() / m.sum().clamp(min=1)
    ratio = torch.exp(logp_new - logp_old)                       # pi_new / pi_old，在 log 空间相减
    pg1 = ratio * advantages
    pg2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -mmean(torch.min(pg1, pg2))                    # 最大化 => 取负号
    if vf_clip is not None and values_old is not None:           # 可选：value clipping
        v_clip = values_old + torch.clamp(values - values_old, -vf_clip, vf_clip)
        vf = torch.max((values - returns) ** 2, (v_clip - returns) ** 2)
    else:
        vf = (values - returns) ** 2
    value_loss = 0.5 * mmean(vf)
    ent = mmean(entropy) if entropy is not None else torch.zeros(())
    total = policy_loss + vf_coef * value_loss - ent_coef * ent   # 熵是奖励项，所以减去
    with torch.no_grad():
        clipfrac = mmean(((ratio - 1).abs() > clip_eps).float())
        approx_kl = mmean((ratio - 1) - torch.log(ratio))        # k3 估计 KL(old || new)
    return total, dict(policy=policy_loss.item(), value=value_loss.item(),
                       entropy=ent.item(), clipfrac=clipfrac.item(), approx_kl=approx_kl.item())

def entropy_from_logits(logits):                                  # (B, T, V) -> (B, T)
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)
# ==== TESTS ====
torch.manual_seed(0)
B, T = 4, 6
lo = torch.randn(B, T) - 2
mask = torch.ones(B, T); mask[0, -2:] = 0
adv = torch.randn(B, T); ret = torch.randn(B, T)
# 1) ratio == 1（刚 rollout 完）时 policy loss = -mean(A)，梯度 = -A * dlogp
ln = lo.clone().requires_grad_(True)
v = torch.zeros(B, T, requires_grad=True)
total, info = ppo_loss(ln, lo, adv, v, ret, mask, ent_coef=0.0)
assert abs(info["policy"] + (adv * mask).sum().item() / mask.sum().item()) < 1e-6
total.backward()
assert torch.allclose(ln.grad, -adv * mask / mask.sum(), atol=1e-6)
# 2) 被 clip 的 token 没有梯度：A>0 且 ratio>1+eps
ln2 = (lo + 0.5).clone().requires_grad_(True)                    # ratio = e^0.5 ≈ 1.65
pos_adv = torch.ones(B, T)
total2, info2 = ppo_loss(ln2, lo, pos_adv, torch.zeros(B, T), torch.zeros(B, T), mask, ent_coef=0.0)
total2.backward()
assert torch.allclose(ln2.grad, torch.zeros(B, T)) and abs(info2["clipfrac"] - 1.0) < 1e-6
# 3) A<0 且 ratio>1+eps：不 clip（min 取未裁剪项），梯度非零 —— PPO 的悲观下界
ln3 = (lo + 0.5).clone().requires_grad_(True)
ppo_loss(ln3, lo, -pos_adv, torch.zeros(B, T), torch.zeros(B, T), mask, ent_coef=0.0)[0].backward()
assert (ln3.grad[mask.bool()] > 0).all()
e = entropy_from_logits(torch.zeros(1, 1, 8)); assert torch.allclose(e, torch.log(torch.tensor(8.0)))
print("ok")
