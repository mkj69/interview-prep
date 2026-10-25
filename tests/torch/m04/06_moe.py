import torch, torch.nn as nn, torch.nn.functional as F
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))     # gain only, no bias, no mean-centering

    def forward(self, x):
        xf = x.float()                                # do the reduction in fp32 even under bf16
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d, hidden=None, multiple_of=256):
        super().__init__()
        hidden = hidden or int(8 * d / 3)             # 8/3 d keeps params ~= a 4d two-matrix FFN
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)  # round up for hardware
        self.w1 = nn.Linear(d, hidden, bias=False)    # gate projection
        self.w3 = nn.Linear(d, hidden, bias=False)    # up projection
        self.w2 = nn.Linear(hidden, d, bias=False)    # down projection

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
class MoE(nn.Module):
    """Token-choice top-k MoE FFN with Switch-style balance loss and router z-loss."""
    def __init__(self, d, n_experts=8, k=2, hidden=None):
        super().__init__()
        self.k, self.E = k, n_experts
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d, hidden) for _ in range(n_experts)])

    def forward(self, x):
        B, T, D = x.shape
        xf = x.reshape(-1, D)                                    # (N, D), N = B*T tokens
        logits = self.router(xf)                                 # (N, E)
        probs = logits.softmax(-1)
        topv, topi = probs.topk(self.k, dim=-1)                  # (N, k)
        gate = topv / topv.sum(-1, keepdim=True)                 # renormalize over the chosen k
        out = torch.zeros_like(xf)
        for e, expert in enumerate(self.experts):
            tok, slot = (topi == e).nonzero(as_tuple=True)       # tokens routed to expert e
            if tok.numel() == 0:
                continue
            out.index_add_(0, tok, gate[tok, slot].unsqueeze(-1) * expert(xf[tok]))
        # f_e: fraction of routing assignments to e (not differentiable); P_e: mean router prob (differentiable)
        counts = torch.zeros(self.E, device=x.device).scatter_add_(
            0, topi.flatten(), torch.ones(topi.numel(), device=x.device))
        f = counts / topi.numel()
        P = probs.mean(0)
        aux_loss = self.E * (f * P).sum()                        # == 1 when perfectly uniform
        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()   # keeps router logits small
        return out.view(B, T, D), aux_loss, z_loss

torch.manual_seed(0)
moe = MoE(32, n_experts=4, k=2, hidden=64)
x = torch.randn(3, 5, 32)
y, aux, z = moe(x)
assert y.shape == x.shape and aux.item() > 0 and z.item() > 0
# reference: dense loop
xf = x.reshape(-1, 32); p = moe.router(xf).softmax(-1); tv, ti = p.topk(2, -1); g = tv / tv.sum(-1, keepdim=True)
ref = torch.stack([sum(g[n, j] * moe.experts[ti[n, j]](xf[n]) for j in range(2)) for n in range(xf.size(0))])
assert torch.allclose(y.reshape(-1, 32), ref, atol=1e-5)
(aux + z).backward()
assert moe.router.weight.grad is not None
print("06 moe ok")
