import math
import torch
import torch.nn.functional as F

def bt_reward_loss(r_chosen, r_rejected):
    # -log sigma(r_c - r_r); logsigmoid is numerically stable
    return -F.logsigmoid(r_chosen - r_rejected).mean()

def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1):
    """Inputs: SUMMED log-probs of each full response, shape (B,)."""
    logits = beta * ((pi_c - ref_c) - (pi_r - ref_r))
    loss = -F.logsigmoid(logits).mean()
    chosen_reward = beta * (pi_c - ref_c).detach()       # implicit rewards, for logging
    rejected_reward = beta * (pi_r - ref_r).detach()
    return loss, chosen_reward, rejected_reward

if __name__ == "__main__":
    pc = torch.tensor([-10.0], requires_grad=True); pr = torch.tensor([-12.0], requires_grad=True)
    loss, cr, rr = dpo_loss(pc, pr, torch.tensor([-11.0]), torch.tensor([-11.5]))
    loss.backward()
    assert pc.grad.item() < 0 < pr.grad.item()           # descent: raise chosen, lower rejected
    m = 0.1 * ((-10 + 11) - (-12 + 11.5)); s = 1 / (1 + math.exp(m))
    assert abs(pc.grad.item() + 0.1 * s) < 1e-6
    l0, _, _ = dpo_loss(torch.tensor([-5.0]), torch.tensor([-7.0]), torch.tensor([-5.0]), torch.tensor([-7.0]))
    assert abs(l0.item() - math.log(2)) < 1e-6            # policy == ref -> log 2
    assert abs(bt_reward_loss(torch.tensor([1.0]), torch.tensor([0.2])).item() - 0.37110) < 1e-4
    print("ok")
