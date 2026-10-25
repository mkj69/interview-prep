import torch

def ppo_loss(logp, old_logp, adv, values, old_values, returns, mask,
             clip_eps=0.2, vf_clip=0.2, vf_coef=0.5, ent=None, ent_coef=0.0):
    """All tensors (B, T) over response tokens; mask=1 for real response tokens."""
    m = mask.float(); n = m.sum().clamp(min=1)
    ratio = torch.exp(logp - old_logp)                       # pi_theta / pi_old, per token
    pg1 = ratio * adv
    pg2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    policy_loss = -(torch.min(pg1, pg2) * m).sum() / n       # maximise -> minimise negative
    v_clipped = old_values + torch.clamp(values - old_values, -vf_clip, vf_clip)
    vf_loss = 0.5 * (torch.max((values - returns) ** 2, (v_clipped - returns) ** 2) * m).sum() / n
    loss = policy_loss + vf_coef * vf_loss
    if ent is not None:
        loss = loss - ent_coef * (ent * m).sum() / n
    with torch.no_grad():
        clipfrac = (((ratio - 1).abs() > clip_eps).float() * m).sum() / n
        approx_kl = (((ratio - 1) - (logp - old_logp)) * m).sum() / n   # k3 estimate of KL(old||new)
    return loss, dict(policy=policy_loss.item(), vf=vf_loss.item(), clipfrac=clipfrac.item(), kl=approx_kl.item())

def kl_shaped_rewards(score, logp_old, logp_ref, mask, beta=0.05):
    """InstructGPT-style: per-token reward = -beta*(log pi_old - log pi_ref); RM score added at last token."""
    r = -beta * (logp_old - logp_ref) * mask
    last = mask.long().cumsum(-1).argmax(-1)                 # index of last real token per row
    r[torch.arange(r.size(0)), last] += score
    return r

if __name__ == "__main__":
    # numeric check of clipping cases on single tokens
    one = torch.ones(1, 1)
    def pol(ratio, A):
        logp = torch.log(torch.tensor([[ratio]], requires_grad=True))
        lp = logp.detach().clone().requires_grad_(True)
        loss, _ = ppo_loss(lp, torch.zeros(1, 1), torch.tensor([[A]]), torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1), one, vf_coef=0.0)
        loss.backward(); return loss.item(), lp.grad.item()
    l, g = pol(1.5, 1.0); assert abs(l + 1.2) < 1e-6 and g == 0.0      # clipped: no gradient
    l, g = pol(0.5, 1.0); assert abs(l + 0.5) < 1e-6 and g < 0          # pushes logp up
    l, g = pol(1.5, -1.0); assert abs(l - 1.5) < 1e-6 and g > 0         # pushes logp down
    l, g = pol(0.5, -1.0); assert abs(l - 0.8) < 1e-6 and g == 0.0      # clipped
    r = kl_shaped_rewards(torch.tensor([2.0, 3.0]), torch.zeros(2, 4), torch.zeros(2, 4), torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1.]]))
    assert r[0, 2] == 2.0 and r[1, 3] == 3.0 and r.sum() == 5.0
    print("ok")
