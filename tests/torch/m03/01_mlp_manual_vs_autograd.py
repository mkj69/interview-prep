import torch, torch.nn.functional as F
torch.manual_seed(0)
B, d, h, C = 8, 5, 16, 3
X = torch.randn(B, d, dtype=torch.float64); y = torch.randint(0, C, (B,))
W1 = torch.randn(d, h, dtype=torch.float64, requires_grad=True); b1 = torch.zeros(h, dtype=torch.float64, requires_grad=True)
W2 = torch.randn(h, C, dtype=torch.float64, requires_grad=True); b2 = torch.zeros(C, dtype=torch.float64, requires_grad=True)
H = X @ W1 + b1; A = H.relu(); Z = A @ W2 + b2
loss = F.cross_entropy(Z, y); loss.backward()
with torch.no_grad():
    dZ = (Z.softmax(-1) - F.one_hot(y, C).double()) / B
    dW2 = A.T @ dZ; db2 = dZ.sum(0)
    dH = (dZ @ W2.T) * (H > 0)
    dW1 = X.T @ dH; db1 = dH.sum(0)
for m, a in ((dW1, W1.grad), (db1, b1.grad), (dW2, W2.grad), (db2, b2.grad)):
    assert torch.allclose(m, a, atol=1e-10)
print("ok")
