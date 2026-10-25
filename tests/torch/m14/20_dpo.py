import torch
import torch.nn.functional as F
# ==== SOLUTION ====
def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    """
    四个输入都是 (B,)：整条 response 的 log prob 之和（只对 response token 求和）。
    ref_* 来自冻结的参考模型（no_grad）。
    L = -log sigmoid( beta * [(pi_c - ref_c) - (pi_r - ref_r)] )
    """
    chosen_reward = beta * (pi_chosen - ref_chosen)        # 隐式 reward
    rejected_reward = beta * (pi_rejected - ref_rejected)
    logits = chosen_reward - rejected_reward
    loss = -F.logsigmoid(logits).mean()                    # 用 logsigmoid，不要 log(sigmoid(.))
    acc = (logits > 0).float().mean()
    return loss, chosen_reward.detach(), rejected_reward.detach(), acc

def sequence_logprob(logits, input_ids, resp_mask):
    """logits (B,T,V), input_ids (B,T), resp_mask (B,T-1) -> (B,) response 部分 log prob 之和。"""
    logp = torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    return (logp * resp_mask).sum(-1)
# ==== TESTS ====
torch.manual_seed(0)
pc, pr, rc, rr = torch.randn(4, 8)
loss, cr, rj, acc = dpo_loss(pc, pr, rc, rr, beta=0.1)
z = 0.1 * ((pc - rc) - (pr - rr))
assert torch.allclose(loss, torch.log1p(torch.exp(-z)).mean(), atol=1e-6)
# 策略 == 参考模型时，loss = log 2
l0 = dpo_loss(rc, rr, rc, rr)[0]; assert abs(l0.item() - torch.log(torch.tensor(2.0)).item()) < 1e-6
# 梯度方向：提高 chosen、降低 rejected
pc_ = pc.clone().requires_grad_(True); pr_ = pr.clone().requires_grad_(True)
dpo_loss(pc_, pr_, rc, rr)[0].backward()
assert (pc_.grad < 0).all() and (pr_.grad > 0).all()
# 极端 logits 下数值稳定
assert torch.isfinite(dpo_loss(torch.tensor([-1e4]), torch.tensor([1e4]), torch.zeros(1), torch.zeros(1))[0])
print("ok")
