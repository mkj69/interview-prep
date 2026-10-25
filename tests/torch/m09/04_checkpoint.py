# Activation checkpointing: same loss and gradients, fewer saved activations. Runs on CPU.
import torch
from torch.utils.checkpoint import checkpoint


class Block(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d)
        self.ff = torch.nn.Sequential(torch.nn.Linear(d, 4 * d), torch.nn.GELU(), torch.nn.Linear(4 * d, d))

    def forward(self, x):
        return x + self.ff(self.norm(x))


class Net(torch.nn.Module):
    def __init__(self, d=32, n=4, ckpt=False):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block(d) for _ in range(n)])
        self.ckpt = ckpt

    def forward(self, x):
        for blk in self.blocks:
            if self.ckpt and self.training:
                # only the block input is saved; the block's internals are recomputed in backward
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x


torch.manual_seed(0)
a = Net(ckpt=False)
b = Net(ckpt=True); b.load_state_dict(a.state_dict())
x = torch.randn(8, 16, 32)
la = a(x).pow(2).mean(); la.backward()
lb = b(x).pow(2).mean(); lb.backward()
assert torch.allclose(la, lb)
for p, q in zip(a.parameters(), b.parameters()):
    assert torch.allclose(p.grad, q.grad, atol=1e-6)
print("checkpoint OK")
