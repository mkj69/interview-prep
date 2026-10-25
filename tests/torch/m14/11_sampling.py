import torch
# ==== SOLUTION ====
def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    """
    logits: (B, V) —— 最后一个位置的 logits。返回 (B,) long。
    顺序：temperature -> top-k -> top-p -> softmax -> 采样。temperature == 0 表示 greedy。
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]           # (B, 1) 第 k 大的值
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        cum = probs.cumsum(dim=-1)
        # 去掉 "在它之前的累计概率已经 >= top_p" 的 token；保证至少保留第 1 个
        remove = (cum - probs) >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
# ==== TESTS ====
g = torch.Generator().manual_seed(0)
logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
assert sample_next(logits, temperature=0.0).item() == 0
draws = torch.stack([sample_next(logits, top_k=2, generator=g) for _ in range(4000)])
assert set(draws.unique().tolist()) == {0, 1}
assert abs((draws == 0).float().mean().item() - 0.5 / 0.8) < 0.03          # 重新归一化
# top_p=0.7: 0.5 < 0.7 -> 需要 {0,1}（累计 0.8）；token 2 之前累计 0.8 >= 0.7 被去掉
draws = torch.stack([sample_next(logits, top_p=0.7, generator=g) for _ in range(4000)])
assert set(draws.unique().tolist()) == {0, 1}
draws = torch.stack([sample_next(logits, top_p=0.01, generator=g) for _ in range(200)])
assert set(draws.unique().tolist()) == {0}                                   # 至少保留 1 个
# 低温 -> 接近 greedy
draws = torch.stack([sample_next(logits, temperature=0.05, generator=g) for _ in range(500)])
assert (draws == 0).float().mean() > 0.99
# batch 维度正确
assert sample_next(torch.randn(5, 10), top_k=3, top_p=0.9, generator=g).shape == (5,)
print("ok")
