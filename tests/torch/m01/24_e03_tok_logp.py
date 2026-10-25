import torch

def masked_token_logp(logits, targets, mask):
    """
    logits:  (B, T, V) float
    targets: (B, T)    long
    mask:    (B, T)    bool，True = 计入（例如 response token）
    返回: 每条序列的 log-prob 之和 (B,) 和 全部有效 token 的平均 log-prob (标量)
    """
    logp = logits.log_softmax(-1)                                  # (B, T, V)
    tok = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)       # (B, T)
    tok = tok * mask                                               # 屏蔽位置置 0
    seq_logp = tok.sum(-1)                                         # (B,)
    mean_logp = tok.sum() / mask.sum().clamp_min(1)                # 按 token 平均
    return seq_logp, mean_logp

torch.manual_seed(0)
B, T, V = 2, 4, 7
logits = torch.randn(B, T, V)
targets = torch.randint(0, V, (B, T))
mask = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=torch.bool)
seq, mean = masked_token_logp(logits, targets, mask)
ref = [sum(torch.log_softmax(logits[b, t], -1)[targets[b, t]].item()
           for t in range(T) if mask[b, t]) for b in range(B)]
assert torch.allclose(seq, torch.tensor(ref), atol=1e-5)
assert torch.allclose(mean, seq.sum() / 6, atol=1e-6)
