import torch

x = torch.arange(10)
s = x[2:5];        s[0] = -1;  print(x[2].item())   # -1：基础切片是 view
x = torch.arange(10)
f = x[[2, 3, 4]];  f[0] = -1;  print(x[2].item())   # 2：高级索引（整数列表/张量）是 copy
m = x[x > 5];      m[0] = -1;  print(x[6].item())   # 6：布尔掩码"读"也是 copy
x[x > 5] = 0                                        # 但"写"是原地的（走 __setitem__）
print(x)                                            # tensor([0, 1, 2, 3, 4, 5, 0, 0, 0, 0])
