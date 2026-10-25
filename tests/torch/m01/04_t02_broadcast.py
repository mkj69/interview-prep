import torch

a = torch.randn(8, 1, 6, 1)
b = torch.randn(   7, 1, 5)
print((a + b).shape)                 # torch.Size([8, 7, 6, 5])

# 经典静默 bug：(B, 1) 和 (B,) 相减 → (B, B)
B = 4
pred = torch.randn(B, 1)             # 回归头 nn.Linear(d, 1) 的输出
y = torch.randn(B)                   # 标签
print((pred - y).shape)              # torch.Size([4, 4])   ← 不是 (4,)！
loss_bad = ((pred - y) ** 2).mean()  # 能跑，数值看起来也"合理"，但完全错了
loss_ok = ((pred.squeeze(-1) - y) ** 2).mean()
# F.mse_loss(pred, y) 至少会给 UserWarning: Using a target size ... different to the input size

# keepdim：按行归一化
x = torch.rand(3, 5)
p = x / x.sum(dim=1, keepdim=True)   # (3, 5) / (3, 1)  ✓
assert torch.allclose(p.sum(1), torch.ones(3))
# x / x.sum(dim=1)：(3, 5) / (3,) 末维 5 ≠ 3 → 报错；
# 但如果 x 恰好是方阵 (3, 3)，就会"成功"地按列去除 → 静默错误
