import torch

A, B = torch.randn(2, 3), torch.randn(3, 4)
assert torch.allclose(torch.einsum("ik,kj->ij", A, B), A @ B)                  # matmul
Ab, Bb = torch.randn(5, 2, 3), torch.randn(5, 3, 4)
assert torch.allclose(torch.einsum("bik,bkj->bij", Ab, Bb), torch.bmm(Ab, Bb))  # bmm

q = torch.randn(2, 4, 6, 8)    # (B, H, Tq, Dh)
k = torch.randn(2, 4, 7, 8)    # (B, H, Tk, Dh)
v = torch.randn(2, 4, 7, 8)
scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / 8 ** 0.5        # (B, H, Tq, Tk)
assert torch.allclose(scores, q @ k.transpose(-2, -1) / 8 ** 0.5, atol=1e-5)
out = torch.einsum("bhqk,bhkd->bhqd", scores.softmax(-1), v)      # (B, H, Tq, Dh)

u, w = torch.randn(3), torch.randn(4)
assert torch.allclose(torch.einsum("i,j->ij", u, w), torch.outer(u, w))        # outer
M = torch.randn(4, 4)
assert torch.allclose(torch.einsum("ii->", M), M.trace())                      # trace
assert torch.allclose(torch.einsum("ii->i", M), M.diagonal())                  # 对角线
assert torch.allclose(torch.einsum("ij->ji", A), A.T)                          # 转置
assert torch.allclose(torch.einsum("ij->j", A), A.sum(0))                      # 按列求和
x, W, y = torch.randn(5, 3), torch.randn(3, 4), torch.randn(5, 4)
assert torch.allclose(torch.einsum("bi,ij,bj->b", x, W, y),
                      ((x @ W) * y).sum(-1), atol=1e-5)                        # bilinear
