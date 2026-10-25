import torch

def grpo_advantages(rewards, eps=1e-6, scale_std=True):
    """rewards: (num_prompts, G) scalar reward per completion -> same-shape advantages."""
    mean = rewards.mean(dim=1, keepdim=True)
    adv = rewards - mean
    if scale_std:                                   # Dr. GRPO sets scale_std=False
        adv = adv / (rewards.std(dim=1, keepdim=True, unbiased=False) + eps)
    return adv

def grpo_loss(logp, old_logp, ref_logp, adv, mask, clip_eps=0.2, beta=0.04, token_level=False):
    """logp/old_logp/ref_logp, mask: (N, T) with N = num_prompts*G; adv: (N,) one value per sequence."""
    m = mask.float()
    ratio = torch.exp(logp - old_logp)
    A = adv.unsqueeze(1)                            # broadcast sequence advantage to every token
    surr = torch.min(ratio * A, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * A)
    log_r = ref_logp - logp                         # log(pi_ref / pi_theta)
    kl = torch.exp(log_r) - log_r - 1               # k3 estimator, >= 0 per token
    per_tok = -(surr - beta * kl)
    if token_level:                                 # DAPO: average over all tokens in batch
        return (per_tok * m).sum() / m.sum()
    # original GRPO: mean over each sequence's tokens, then mean over sequences
    return ((per_tok * m).sum(1) / m.sum(1)).mean()

if __name__ == "__main__":
    r = torch.tensor([[1., 0, 0, 1, 1, 0, 0, 0], [1., 1, 1, 1, 1, 1, 1, 1]])
    a = grpo_advantages(r)
    assert torch.allclose(a[0], torch.tensor([1.291, -.775, -.775, 1.291, 1.291, -.775, -.775, -.775]), atol=1e-3)
    assert torch.all(a[1] == 0)                     # all-correct group gives zero gradient
    N, T = 16, 5
    torch.manual_seed(0)
    lp = torch.randn(N, T) * 0.1 - 1
    mask = torch.ones(N, T); mask[0, 3:] = 0
    L = grpo_loss(lp.clone().requires_grad_(True), lp, lp, a.reshape(-1), mask)
    # on-policy first step, pi = pi_old = pi_ref -> ratio 1, kl 0, loss = -mean(adv) ~ 0
    assert abs(L.item() + a.reshape(-1).mean().item()) < 1e-6
    lp2 = lp.clone().requires_grad_(True)
    grpo_loss(lp2, lp, lp, a.reshape(-1), mask, token_level=True).backward()
    assert lp2.grad[0, 0] < 0 and lp2.grad[1, 0] > 0   # positive adv -> raise logp; negative -> lower
    print("ok")
