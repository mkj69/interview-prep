import torch
import torch.nn.functional as F

# scatter_：dim=1 时 out[i][ index[i][j] ] = src —— gather 的"反方向"
labels = torch.tensor([0, 2, 1])
one_hot = torch.zeros(3, 4).scatter_(1, labels.unsqueeze(1), 1.0)
assert torch.equal(one_hot, F.one_hot(labels, num_classes=4).float())

# scatter_add_：重复索引会累加（普通 scatter_ 遇到重复索引时结果不确定）
ids = torch.tensor([0, 1, 1, 3, 3, 3])
counts = torch.zeros(5).scatter_add_(0, ids, torch.ones(6))
print(counts)                         # tensor([1., 2., 0., 3., 0.])
assert torch.equal(counts, torch.bincount(ids, minlength=5).float())

# index_add_：按段求和（例如把 token 级的量汇总到样本级）
vals = torch.tensor([1., 2., 3., 4., 5., 6.])
print(torch.zeros(5).index_add_(0, ids, vals))   # tensor([ 1.,  5.,  0., 15.,  0.])
