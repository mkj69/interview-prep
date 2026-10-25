import torch

T = 4
scores = torch.randn(T, T)                                            # (query, key)
causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)   # True = 未来位置，要屏蔽
print(causal.int())
# tensor([[0, 1, 1, 1],
#         [0, 0, 1, 1],
#         [0, 0, 0, 1],
#         [0, 0, 0, 0]])
att = scores.masked_fill(causal, float("-inf")).softmax(dim=-1)
assert torch.allclose(att.sum(-1), torch.ones(T))
assert (att[causal] == 0).all()

# 等价写法：tril 当作"允许看"的矩阵
allowed = torch.tril(torch.ones(T, T, dtype=torch.bool))
att2 = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
assert torch.allclose(att, att2)

# 整行都被屏蔽 → softmax 得到 NaN（左 padding 的 query 很常见）
print(torch.full((3,), float("-inf")).softmax(-1))   # tensor([nan, nan, nan])

# where：逐元素的三目运算
z = torch.tensor([-1., 2., -3.])
print(torch.where(z > 0, z, torch.zeros_like(z)))    # tensor([0., 2., 0.])
