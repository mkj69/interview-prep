import numpy as np
import torch

print(torch.tensor([1, 2]).dtype)            # torch.int64
print(torch.tensor([1.0]).dtype)             # torch.float32（默认浮点 dtype）
print(torch.tensor(np.array([1.0])).dtype)   # torch.float64 —— 继承了 numpy 的 float64！
print((torch.tensor([3]) / 2).dtype)         # torch.float32：/ 永远是真除法
print((torch.tensor([3]) // 2).dtype)        # torch.int64
print((torch.tensor([3]) + 0.5).dtype)       # torch.float32：Python 标量不会把结果升成 float64

a = np.zeros(3)
t = torch.from_numpy(a)      # 共享内存（零拷贝），dtype 跟随 numpy：float64
t[0] = 5
print(a)                     # [5. 0. 0.]
u = torch.tensor(a)          # torch.tensor(...) 总是复制
u[1] = 7
print(a)                     # [5. 0. 0.]：没变

w = torch.ones(2, requires_grad=True)
# w.numpy()  → RuntimeError: Can't call numpy() on Tensor that requires grad ...
arr = w.detach().cpu().numpy()   # 标准写法：断开计算图 → 搬回 CPU → 转 numpy

device = "cuda" if torch.cuda.is_available() else "cpu"
z = torch.zeros(2)
z2 = z.to(device, torch.bfloat16)   # tensor.to 返回新 tensor（若什么都没变则返回自身）
model = torch.nn.Linear(2, 2)
model.to(device)                    # module.to 原地修改所有参数，并返回 self
