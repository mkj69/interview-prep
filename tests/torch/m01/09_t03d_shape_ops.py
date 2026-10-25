import torch

q = torch.randn(1, 3, 1)
print(q.squeeze().shape)        # torch.Size([3])        ← 危险：batch=1 时连 batch 维也被挤掉
print(q.squeeze(-1).shape)      # torch.Size([1, 3])     ← 推荐：永远写明 dim
print(q.unsqueeze(0).shape)     # torch.Size([1, 1, 3, 1])

base = torch.randn(1, 3)
e = base.expand(1000, 3)        # 不分配新内存，stride (0, 1)
r = base.repeat(1000, 1)        # 真的复制 1000 份
print(e.stride(), r.stride())   # (0, 1) (3, 1)
# e.add_(1) → RuntimeError: more than one element of the written-to tensor refers to a single memory location

a, b = torch.randn(2, 3), torch.randn(2, 3)
print(torch.cat([a, b], dim=0).shape)     # torch.Size([4, 3])     沿已有维拼接
print(torch.stack([a, b], dim=0).shape)   # torch.Size([2, 2, 3])  新建一维再拼
