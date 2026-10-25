import torch

x = torch.tensor(2.0, requires_grad=True)    # 叶子节点（leaf）：用户创建、requires_grad=True
w = torch.tensor(3.0, requires_grad=True)
y = w * x + x ** 2                           # 非叶子：由运算产生，带 grad_fn
print(x.is_leaf, y.is_leaf, y.grad_fn)       # True False <AddBackward0 ...>
y.backward()                                 # y 是标量，可直接 backward
print(x.grad, w.grad)                        # dy/dx = w + 2x = 7,  dy/dw = x = 2

# 梯度是"累加"进 .grad 的，不是覆盖
y = w * x + x ** 2
y.backward()
print(x.grad)                                # tensor(14.)  ← 7 + 7
x.grad = None                                # 清零（optimizer.zero_grad() 默认就是设为 None）

# 非标量输出要么先规约成标量，要么传入 gradient（即向量-雅可比积 VJP 里的向量）
v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
out = v * 2
out.backward(torch.ones_like(out))           # 等价于 out.sum().backward()
print(v.grad)                                # tensor([2., 2., 2.])

# 中间结果默认不保留 .grad；需要时调用 retain_grad()
a = torch.tensor(1.0, requires_grad=True)
h = a * 3
h.retain_grad()
(h ** 2).backward()
print(h.grad, a.grad)                        # tensor(6.) tensor(18.)
