import torch

w = torch.randn(3, requires_grad=True)

y = (w * 2).detach()          # detach：同一块数据，但从图上断开（stop-gradient）
print(y.requires_grad)        # False

with torch.no_grad():         # no_grad：块内的运算不记录计算图（省显存、提速）
    z = w * 2
print(z.requires_grad)        # False

with torch.inference_mode():  # inference_mode：更激进的 no_grad，产出的 tensor 之后不能参与 autograd
    u = w * 2
print(u.is_inference())       # True

# stop-gradient 的典型用途：target 不该接收梯度
pred = w.sum()
target = (w ** 2).sum().detach()   # 例如 TD target、EMA teacher、PPO 里的 old log-prob
loss = (pred - target) ** 2
loss.backward()               # 梯度只流经 pred
