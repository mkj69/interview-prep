import torch

def pairwise_sq_dist(X, Y):
    """X: (N, D), Y: (M, D) -> (N, M)，D[i, j] = ||X[i] - Y[j]||^2"""
    x2 = (X * X).sum(-1, keepdim=True)        # (N, 1)
    y2 = (Y * Y).sum(-1)                      # (M,)  → 广播成 (1, M)
    d2 = x2 - 2 * X @ Y.T + y2                # (N, M)
    return d2.clamp_min(0)                    # 浮点相消可能产生 -1e-7 这样的负数

X, Y = torch.randn(5, 3), torch.randn(7, 3)
naive = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)   # 中间张量 (N, M, D)：显存 O(NMD)
assert torch.allclose(pairwise_sq_dist(X, Y), naive, atol=1e-5)
assert torch.allclose(pairwise_sq_dist(X, Y).sqrt(), torch.cdist(X, Y), atol=1e-4)
