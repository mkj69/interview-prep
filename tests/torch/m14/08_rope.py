import torch
# ==== SOLUTION ====
def rope_cache(T: int, d: int, base: float = 10000.0):
    """返回 cos, sin: (T, d/2)。第 i 对维度的角频率 theta_i = base^(-2i/d)。"""
    assert d % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))    # (d/2,)
    pos = torch.arange(T).float()                                     # (T,)
    angles = torch.outer(pos, inv_freq)                               # (T, d/2)
    return angles.cos(), angles.sin()

def apply_rope(x, cos, sin):
    """
    x: (B, H, T, d)；cos/sin: (T, d/2)。
    使用 "rotate-half" 配对 (LLaMA/HF 约定)：维度 i 和 i + d/2 组成一对。
    """
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    # 二维旋转: (x1, x2) -> (x1 cos - x2 sin, x1 sin + x2 cos)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
# ==== TESTS ====
torch.manual_seed(0)
B, H, T, d = 1, 2, 12, 16
cos, sin = rope_cache(T, d)
q, k = torch.randn(B, H, T, d), torch.randn(B, H, T, d)
qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
assert torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-5)   # 旋转保范数
assert torch.allclose(qr[:, :, 0], q[:, :, 0])                       # 位置 0 不旋转
# 相对位置性质：<R_m q, R_n k> 只取决于 m - n
q1 = q[:, :, :1].expand(B, H, T, d); k1 = k[:, :, :1].expand(B, H, T, d)
q1r, k1r = apply_rope(q1, cos, sin), apply_rope(k1, cos, sin)
s_a = (q1r[:, :, 5] * k1r[:, :, 2]).sum(-1)    # m=5, n=2
s_b = (q1r[:, :, 9] * k1r[:, :, 6]).sum(-1)    # m=9, n=6
assert torch.allclose(s_a, s_b, atol=1e-4)
# 复数形式等价性检查
xc = torch.complex(q[..., : d // 2], q[..., d // 2 :])
rot = torch.polar(torch.ones_like(cos), torch.atan2(sin, cos))
out = xc * rot
assert torch.allclose(torch.cat([out.real, out.imag], -1), qr, atol=1e-5)
print("ok")
