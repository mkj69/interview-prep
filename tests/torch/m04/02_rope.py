import torch
def rope_cache(T, dh, base=10000.0, device=None):
    inv_freq = base ** (-torch.arange(0, dh, 2, device=device).float() / dh)  # theta_i, (dh/2,)
    pos = torch.arange(T, device=device).float()
    ang = torch.outer(pos, inv_freq)                 # (T, dh/2): angle = m * theta_i
    ang = torch.cat([ang, ang], dim=-1)              # (T, dh): "half-split" layout (HF LLaMA)
    return ang.cos(), ang.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)                      # pair dim i with dim i + dh/2
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, cos, sin):
    """x: (B, H, T, dh); cos, sin: (T, dh), broadcast over B and H."""
    return x * cos + rotate_half(x) * sin

def apply_rope_complex(x, base=10000.0):
    """Same rotation in the 'interleaved' layout (pairs (0,1), (2,3), ...), as in Meta's original LLaMA code."""
    *lead, T, dh = x.shape
    inv_freq = base ** (-torch.arange(0, dh, 2).float() / dh)
    ang = torch.outer(torch.arange(T).float(), inv_freq)                 # (T, dh/2)
    rot = torch.polar(torch.ones_like(ang), ang)                         # e^{i m theta}
    xc = torch.view_as_complex(x.float().reshape(*lead, T, dh // 2, 2).contiguous())  # (..., T, dh/2)
    return torch.view_as_real(xc * rot).flatten(-2).type_as(x)

torch.manual_seed(0)
T, dh = 32, 16
cos, sin = rope_cache(T, dh)
qv, kv = torch.randn(dh), torch.randn(dh)
Q = apply_rope(qv.expand(1, 1, T, dh), cos, sin)[0, 0]
K = apply_rope(kv.expand(1, 1, T, dh), cos, sin)[0, 0]
S = Q @ K.T
assert torch.allclose(S[9, 4], S[5, 0], atol=1e-4) and torch.allclose(S[30, 25], S[5, 0], atol=1e-4)
assert torch.allclose(S[7, 7], qv @ kv, atol=1e-4)
assert torch.allclose(Q.norm(dim=-1), qv.norm().expand(T), atol=1e-4)
Qc = apply_rope_complex(qv.expand(1, 1, T, dh))[0, 0]; Kc = apply_rope_complex(kv.expand(1, 1, T, dh))[0, 0]
Sc = Qc @ Kc.T
assert torch.allclose(Sc[20, 3], Sc[17, 0], atol=1e-4)
perm = torch.cat([torch.arange(0, dh, 2), torch.arange(1, dh, 2)])
x = torch.randn(1, 1, T, dh)
assert torch.allclose(apply_rope_complex(x)[..., perm], apply_rope(x[..., perm], cos, sin), atol=1e-5)
print("02 rope ok")
