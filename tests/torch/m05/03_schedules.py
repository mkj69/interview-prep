import math, torch

def cosine_lr(step, total, peak, warmup, min_ratio=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))

def wsd_lr(step, total, peak, warmup, decay_frac=0.1, final=0.0):
    decay_start = int(total * (1 - decay_frac))
    if step < warmup:
        return peak * (step + 1) / warmup
    if step < decay_start:
        return peak
    p = (step - decay_start) / max(1, total - decay_start)
    return final + (peak - final) * (1 - p)

T = 1000
w = torch.nn.Parameter(torch.zeros(3))
opt = torch.optim.AdamW([w], lr=3e-4)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: wsd_lr(s, T, 1.0, 50))
lrs = []
for s in range(T):
    lrs.append(opt.param_groups[0]['lr']); opt.step(); sched.step()
assert abs(lrs[500] - 3e-4) < 1e-12 and lrs[-1] < 1e-6 and lrs[0] < 1e-5
assert abs(cosine_lr(T, T, 3e-4, 50) - 3e-5) < 1e-12
print("ok")
