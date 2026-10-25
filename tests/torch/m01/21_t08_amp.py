import sys, torch
if not torch.cuda.is_available():
    print("SKIP: needs CUDA"); sys.exit(0)
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda"   # 这一段需要 GPU
model = nn.Linear(1024, 1024).to(device)            # 参数（master weights）保持 fp32
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("cuda")               # 只有 fp16 需要；bf16 不需要

x = torch.randn(64, 1024, device=device)
target = torch.randn(64, 1024, device=device)

for step in range(10):
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(x)                              # matmul 在 fp16 下算
        loss = F.mse_loss(out.float(), target)      # loss 用 fp32 更稳（autocast 本身也会把 mse_loss 等放到 fp32）
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()                   # loss × S 再 backward，防止小梯度在 fp16 下溢为 0
    scaler.unscale_(opt)                            # 裁剪前先除回 S，否则阈值不对
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)                                # 若梯度含 inf/NaN 则跳过这一步
    scaler.update()                                 # 动态调整 S：溢出就减半，长期没溢出就增大

# bf16 版本：去掉 scaler 即可
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = F.mse_loss(model(x).float(), target)
loss.backward()
