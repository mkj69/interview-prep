import torch, torch.nn as nn

class VelocityNet(nn.Module):
    def __init__(self, d=2, n_cls=8, h=256):
        super().__init__()
        self.emb = nn.Embedding(n_cls + 1, h)       # last index = "null" label for CFG
        self.null = n_cls
        self.inp = nn.Linear(d + 1, h)
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d))
    def forward(self, x, t, y):
        return self.net(self.inp(torch.cat([x, t], 1)) + self.emb(y))

def rf_loss(model, x1, y, p_uncond=0.1):
    x0 = torch.randn_like(x1)                       # noise
    t = torch.rand(x1.size(0), 1)                   # t ~ U[0,1]  (SD3 uses logit-normal)
    xt = t * x1 + (1 - t) * x0                      # straight-line interpolation
    drop = torch.rand(y.shape) < p_uncond           # label dropout -> trains p(x) too
    y = torch.where(drop, torch.full_like(y, model.null), y)
    return ((model(xt, t, y) - (x1 - x0)) ** 2).sum(1).mean()

@torch.no_grad()
def sample(model, y, n_steps=16, w=2.0):
    x = torch.randn(y.size(0), 2)
    null = torch.full_like(y, model.null)
    for i in range(n_steps):
        t = torch.full((y.size(0), 1), i / n_steps)
        v_c, v_u = model(x, t, y), model(x, t, null)
        v = v_u + w * (v_c - v_u)                   # CFG on velocity (w=1: no guidance)
        x = x + v / n_steps                         # Euler step
    return x

if __name__ == '__main__':
    torch.manual_seed(0)
    ang = 2 * torch.pi * torch.arange(8) / 8
    C = 4 * torch.stack([ang.cos(), ang.sin()], 1)
    model = VelocityNet(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(3000):
        y = torch.randint(0, 8, (512,))
        x1 = C[y] + 0.3 * torch.randn(512, 2)
        loss = rf_loss(model, x1, y)
        opt.zero_grad(); loss.backward(); opt.step()
    y = torch.randint(0, 8, (2000,))
    xs = sample(model, y, n_steps=32, w=2.0)
    acc = ((xs - C[y]).norm(dim=1) < 1.0).float().mean()
    assert acc > 0.8, acc
    print('OK', float(acc))
