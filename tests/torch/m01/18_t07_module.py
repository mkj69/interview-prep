import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, p_drop=0.1):
        super().__init__()                         # 必须先调用，否则注册参数会报错
        self.fc1 = nn.Linear(d_in, d_hidden)       # 子模块赋值给属性 → 自动注册其参数
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.drop = nn.Dropout(p_drop)
        self.scale = nn.Parameter(torch.ones(d_out))            # 可训练参数
        self.register_buffer("step", torch.zeros((), dtype=torch.long))  # 非训练状态：随 .to() 搬运、存进 state_dict

    def forward(self, x):                          # 调用 model(x)，不要直接调 model.forward(x)
        h = self.drop(torch.relu(self.fc1(x)))
        return self.fc2(h) * self.scale

model = MLP(16, 64, 4)
for name, p in model.named_parameters():      # 顺序：模块自己的参数在前，再按子模块注册顺序
    print(name, tuple(p.shape))
# scale (4,)
# fc1.weight (64, 16)   ← nn.Linear 的 weight 形状是 (out, in)，计算 y = x @ W.T + b
# fc1.bias (64,)
# fc2.weight (4, 64)
# fc2.bias (4,)
n_params = sum(p.numel() for p in model.parameters())
print(n_params)                                     # 1352 = (16*64+64) + (64*4+4) + 4
assert n_params == 1352
assert "step" in model.state_dict()                 # buffer 在 state_dict 里……
assert all(n != "step" for n, _ in model.named_parameters())   # ……但不在 parameters() 里

torch.save(model.state_dict(), "mlp.pt")            # 推荐：只存 state_dict
model2 = MLP(16, 64, 4)
model2.load_state_dict(torch.load("mlp.pt"))        # 先构造同结构模型，再加载
model2.eval()                                       # 推理前切到 eval：关闭 dropout，BN 用 running stats
