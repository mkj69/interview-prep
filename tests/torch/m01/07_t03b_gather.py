import torch
import torch.nn.functional as F

# 小例子：dim=1 时 out[i][j] = src[i][ index[i][j] ]
src = torch.tensor([[10, 11, 12],
                    [20, 21, 22]])
idx = torch.tensor([[2, 0],
                    [1, 1]])
print(torch.gather(src, 1, idx))     # [[12, 10], [21, 21]]

# LLM 里最常用：取出每个位置上"目标 token"的 log-prob
B, T, V = 2, 3, 5
logits = torch.randn(B, T, V)
targets = torch.randint(0, V, (B, T))
logp = logits.log_softmax(dim=-1)                                # (B, T, V)
tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)    # (B, T)

# 等价写法：高级索引（两个辅助下标广播成 (B, T)）
b = torch.arange(B)[:, None]
t = torch.arange(T)[None, :]
assert torch.allclose(tok_logp, logp[b, t, targets])
# 和 cross_entropy 的关系：CE = -mean(tok_logp)
assert torch.allclose(-tok_logp.mean(),
                      F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1)))
