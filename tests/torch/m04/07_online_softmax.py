import torch
torch.manual_seed(0)
x = torch.randn(1000, dtype=torch.float64) * 30
m, l = torch.tensor(float('-inf'), dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)
for xi in x:
    m_new = torch.maximum(m, xi)
    l = l * torch.exp(m - m_new) + torch.exp(xi - m_new)
    m = m_new
assert torch.allclose(torch.exp(x - m) / l, torch.softmax(x, 0))
print("07 online softmax ok")
