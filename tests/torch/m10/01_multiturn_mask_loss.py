import torch

def build_loss_mask(roles):
    """roles: list[str] per token, one of 'prompt','assistant','tool'.
    只在模型自己生成的 token (assistant) 上算 loss；tool 输出和 prompt 被 mask 掉。"""
    return torch.tensor([r == "assistant" for r in roles], dtype=torch.float32)

def multiturn_pg_loss(logprobs, loss_mask, advantages):
    """logprobs: (B, T) 采样 token 在当前策略下的 log-prob
    loss_mask: (B, T) 1=assistant token
    advantages: (B,) 轨迹级 advantage（如 GRPO 组内标准化后的 reward）
    返回 token-level 平均的 REINFORCE 型 loss（on-policy，无 ratio/clip）。"""
    per_tok = -advantages[:, None] * logprobs * loss_mask          # (B, T)
    return per_tok.sum() / loss_mask.sum().clamp(min=1.0)

def group_advantages(rewards, eps=1e-6):
    """GRPO 风格：同一任务的 G 条轨迹，reward 减均值除以标准差。"""
    return (rewards - rewards.mean()) / (rewards.std() + eps)

if __name__ == "__main__":
    roles = ["prompt"] * 3 + ["assistant"] * 4 + ["tool"] * 5 + ["assistant"] * 2
    m = build_loss_mask(roles)
    assert m.sum().item() == 6 and m[3:7].sum() == 4 and m[7:12].sum() == 0

    torch.manual_seed(0)
    B, T = 4, len(roles)
    logits = torch.randn(B, T, 10, requires_grad=True)
    tokens = torch.randint(0, 10, (B, T))
    logprobs = torch.log_softmax(logits, -1).gather(-1, tokens[..., None]).squeeze(-1)
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
    adv = group_advantages(rewards)
    assert torch.allclose(adv.sum(), torch.tensor(0.0), atol=1e-5)
    mask = m.expand(B, T)
    loss = multiturn_pg_loss(logprobs, mask, adv)
    loss.backward()
    # tool / prompt 位置不应收到任何梯度
    g = logits.grad.abs().sum(-1)            # (B, T)
    assert torch.all(g[:, 7:12] == 0) and torch.all(g[:, :3] == 0)
    assert torch.all(g[:, 3:7] > 0)
    print("loss", float(loss), "ok")
