import torch, torch.nn.functional as F
z = torch.tensor([[1000., 1001., 1002.]])
assert torch.isfinite(F.log_softmax(z, -1)).all()
assert torch.isfinite(F.cross_entropy(z, torch.tensor([0])))
# naive is broken
assert not torch.isfinite(torch.log(torch.exp(z) / torch.exp(z).sum())).all()
# BCE with logits stable
zl = torch.tensor([800., -800.]); yl = torch.tensor([0., 1.])
l = F.binary_cross_entropy_with_logits(zl, yl); assert torch.isfinite(l) and abs(l.item() - 800) < 1e-3
# label smoothing equivalence
torch.manual_seed(0)
logits = torch.randn(4, 5); y = torch.randint(0, 5, (4,)); eps = 0.1
q = torch.full((4, 5), eps / 5); q[torch.arange(4), y] += 1 - eps
ref = -(q * F.log_softmax(logits, -1)).sum(-1).mean()
assert torch.allclose(F.cross_entropy(logits, y, label_smoothing=eps), ref, atol=1e-6)
print("ok")
