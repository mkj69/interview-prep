import torch

# 1) 叶子张量不能原地修改（除非在 no_grad 里，比如优化器更新）
w = torch.ones(3, requires_grad=True)
try:
    w.add_(1)
except RuntimeError as err:
    print("leaf:", err)        # a leaf Variable that requires grad is being used in an in-place operation.
with torch.no_grad():
    w -= 0.1 * torch.ones(3)   # 手写 SGD 更新就这样写

# 2) 原地修改了 backward 需要的值 → backward 时报错（autograd 用 version counter 检测）
a = torch.randn(3, requires_grad=True)
b = a.sigmoid()                # sigmoid 的 backward 需要它的输出 b：σ' = b(1-b)
b.mul_(2)                      # 原地改掉了 b
try:
    b.sum().backward()
except RuntimeError as err:
    print("version:", str(err)[:90])   # one of the variables needed for gradient computation has been modified by an inplace operation

# 3) retain_graph：默认 backward 后释放图里保存的中间结果，第二次 backward 会报错
x = torch.tensor(2.0, requires_grad=True)
y = x.exp()
y.backward(retain_graph=True)
y.backward()                   # 因为上面保留了图，所以这次可以
print(x.grad)                  # 2 * e^2 ≈ 14.78
