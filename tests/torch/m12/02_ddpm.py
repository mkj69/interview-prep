import torch, torch.nn as nn

T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
abar = torch.cumprod(alphas, dim=0)

def q_sample(x0, t, eps):
    """x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps ;  t: (B,) long"""
    a = abar[t].view(-1, *([1] * (x0.dim() - 1)))
    return a.sqrt() * x0 + (1 - a).sqrt() * eps

def ddpm_loss(model, x0):
    t = torch.randint(0, T, (x0.size(0),))
    eps = torch.randn_like(x0)
    xt = q_sample(x0, t, eps)
    return ((model(xt, t) - eps) ** 2).mean()      # L_simple

@torch.no_grad()
def ddpm_sample(model, shape):
    x = torch.randn(shape)
    for t in reversed(range(T)):
        tt = torch.full((shape[0],), t, dtype=torch.long)
        eps = model(x, tt)
        mean = (x - betas[t] / (1 - abar[t]).sqrt() * eps) / alphas[t].sqrt()
        if t > 0:
            x = mean + betas[t].sqrt() * torch.randn_like(x)   # sigma_t^2 = beta_t
        else:
            x = mean
    return x

@torch.no_grad()
def ddim_sample(model, shape, n_steps=50, eta=0.0):
    ts = torch.linspace(T - 1, 0, n_steps).long()
    x = torch.randn(shape)
    for i, t in enumerate(ts):
        tt = torch.full((shape[0],), int(t), dtype=torch.long)
        eps = model(x, tt)
        a_t = abar[t]
        a_prev = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        x0_hat = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        sigma = eta * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).sqrt()
        dir_xt = (1 - a_prev - sigma ** 2).sqrt() * eps
        x = a_prev.sqrt() * x0_hat + dir_xt + sigma * torch.randn_like(x)
    return x

class GaussianOracle(nn.Module):
    """Exact eps-predictor when data ~ N(m, s^2): E[eps | x_t]."""
    def __init__(self, m, s): super().__init__(); self.m, self.s = m, s
    def forward(self, x, t):
        a = abar[t].view(-1, 1)
        var = a * self.s ** 2 + (1 - a)
        return (1 - a).sqrt() * (x - a.sqrt() * self.m) / var

if __name__ == '__main__':
    torch.manual_seed(0)
    oracle = GaussianOracle(2.0, 0.5)
    for sampler, tol in [(lambda: ddpm_sample(oracle, (20000, 1)), 0.03),
                         (lambda: ddim_sample(oracle, (20000, 1), 50), 0.04)]:
        x = sampler()
        assert abs(x.mean() - 2.0) < 0.03 and abs(x.std() - 0.5) < tol, (x.mean(), x.std())
    # training loss runs
    net = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 1))
    model = lambda x, t: net(torch.cat([x, t.float().view(-1, 1) / T], 1))
    loss = ddpm_loss(model, 2.0 + 0.5 * torch.randn(128, 1)); loss.backward()
    assert torch.isfinite(loss)
    print('OK')
