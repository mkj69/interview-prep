import torch, torch.nn as nn, torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, d_in=784, d_h=256, d_z=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU())
        self.mu = nn.Linear(d_h, d_z)
        self.logvar = nn.Linear(d_h, d_z)          # predict log sigma^2 for stability
        self.dec = nn.Sequential(nn.Linear(d_z, d_h), nn.ReLU(), nn.Linear(d_h, d_in))

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        eps = torch.randn_like(mu)                  # noise is outside the graph
        z = mu + torch.exp(0.5 * logvar) * eps      # reparameterization trick
        return self.dec(z), mu, logvar              # decoder returns logits

def vae_loss(x, logits, mu, logvar, beta=1.0):
    # -E_q[log p(x|z)] with Bernoulli decoder, summed over pixels, averaged over batch
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction='sum') / x.size(0)
    # KL(N(mu, sigma^2) || N(0, I)) closed form
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar) / x.size(0)
    return rec + beta * kl, rec, kl

if __name__ == '__main__':
    torch.manual_seed(0)
    x = (torch.rand(64, 784) > 0.5).float()
    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = None
    for step in range(200):
        logits, mu, logvar = model(x)
        loss, rec, kl = vae_loss(x, logits, mu, logvar)
        opt.zero_grad(); loss.backward(); opt.step()
        if first is None: first = loss.item()
    assert loss.item() < first, (first, loss.item())
    assert kl.item() >= 0
    # KL closed form check vs MC
    mu = torch.tensor([[0.5, -1.0]]); logvar = torch.log(torch.tensor([[0.49, 1.69]]))
    kl_cf = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar)
    q = torch.distributions.Normal(mu, torch.exp(0.5 * logvar)); p = torch.distributions.Normal(0., 1.)
    assert torch.allclose(kl_cf, torch.distributions.kl_divergence(q, p).sum(), atol=1e-5)
    print('OK')
