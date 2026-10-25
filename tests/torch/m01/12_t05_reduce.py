import torch

x = torch.tensor([[1., 5., 3.],
                  [7., 2., 2.]])
print(x.sum(dim=0))            # tensor([8., 7., 5.])   沿 dim 0 求和 → 这一维消失
print(x.mean(dim=-1))          # tensor([3.0000, 3.6667])
vals, idx = x.max(dim=1)       # 带 dim 的 max 返回 (values, indices)
print(vals, idx)               # tensor([5., 7.]) tensor([1, 0])
print(x.argmax(dim=1))         # tensor([1, 0])

big = torch.tensor([1000., 1001., 1002.])
print(torch.exp(big) / torch.exp(big).sum())   # tensor([nan, nan, nan])   inf / inf
print(torch.softmax(big, dim=-1))              # tensor([0.0900, 0.2447, 0.6652])
print(torch.logsumexp(big, dim=0))             # tensor(1002.4076)

z = torch.tensor([0., -200.])
print(torch.log(torch.softmax(z, -1)))         # tensor([0., -inf])   exp(-200) 在 fp32 下溢成 0
print(torch.log_softmax(z, -1))                # tensor([   0., -200.])
