import torch

x = torch.arange(12).reshape(3, 4)
t = x.t()                                # (4, 3)，stride (1, 4)，非连续
try:
    t.view(12)
except RuntimeError as err:
    print("view failed:", err)           # view size is not compatible with input tensor's size and stride ...
r = t.reshape(12)                        # view 不行时，reshape 会悄悄复制一份
print(r)                                 # tensor([ 0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11])
print(r.data_ptr() == x.data_ptr())      # False → 这是拷贝
c = t.contiguous()                       # 按当前逻辑顺序重新排成连续内存（拷贝）
print(c.stride(), c.view(12).shape)      # (3, 1) torch.Size([12])

v = x.reshape(4, 3)                      # x 本来就连续 → reshape 返回 view
v[0, 0] = 100
print(x[0, 0].item())                    # 100：改 v 就是改 x

# 多头注意力里最常见的 shape 变换
B, T, H, Dh = 2, 5, 4, 8
q = torch.randn(B, T, H * Dh)
q = q.view(B, T, H, Dh).transpose(1, 2)                  # (B, H, T, Dh)，非连续
out = q.transpose(1, 2).contiguous().view(B, T, H * Dh)  # 合并 head 前必须 contiguous（或用 reshape）
assert out.shape == (B, T, H * Dh)
