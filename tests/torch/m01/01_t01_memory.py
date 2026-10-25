import torch

x = torch.arange(12).reshape(3, 4)       # int64，存在一块连续的 storage 里
print(x.shape, x.dtype, x.device)        # torch.Size([3, 4]) torch.int64 cpu
print(x.stride(), x.storage_offset())    # (4, 1) 0   —— 单位是"元素"，不是字节
print(x.is_contiguous())                 # True

# 元素 (i, j) 在 storage 中的下标 = offset + i*stride[0] + j*stride[1]
i, j = 2, 1
flat = x.flatten()
assert x[i, j] == flat[x.storage_offset() + i * x.stride(0) + j * x.stride(1)]

t = x.t()                                # 转置：不搬数据，只交换 shape 和 stride
print(t.shape, t.stride())               # torch.Size([4, 3]) (1, 4)
print(t.is_contiguous())                 # False
print(t.data_ptr() == x.data_ptr())      # True：同一块内存

s = x[1:, ::2]                           # 切片也是 view：只改 offset 和 stride
print(s.stride(), s.storage_offset())    # (4, 2) 4

e = torch.tensor([1, 2, 3]).expand(4, 3) # expand：新维度 stride = 0，"假装"重复
print(e.stride())                        # (0, 1)
