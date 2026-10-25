import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def sdpa(q, k, v, causal=True, key_padding_mask=None):
    """q: (B, H, Tq, dh); k, v: (B, H, Tk, dh); key_padding_mask: (B, Tk) bool, True = real token."""
    dh = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(dh)            # (B, H, Tq, Tk)
    Tq, Tk = scores.shape[-2:]
    if causal:
        # query i sits at absolute position Tk - Tq + i and may see keys j <= that position.
        # (bottom-right alignment: also correct when decoding with a KV cache, Tq < Tk)
        allowed = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(diagonal=Tk - Tq)
        scores = scores.masked_fill(~allowed, float('-inf'))
    if key_padding_mask is not None:
        scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float('-inf'))
    attn = scores.softmax(dim=-1)                                 # normalize over KEYS (last dim)
    return attn @ v, attn                                         # (B, H, Tq, dh)
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
class Attention(nn.Module):
    """Causal self-attention. n_kv_heads = n_heads -> MHA; 1 -> MQA; in between -> GQA."""
    def __init__(self, d_model, n_heads, n_kv_heads=None):
        super().__init__()
        self.h = n_heads
        self.h_kv = n_kv_heads or n_heads
        assert d_model % n_heads == 0 and n_heads % self.h_kv == 0
        self.dh = d_model // n_heads
        self.wq = nn.Linear(d_model, self.h * self.dh, bias=False)
        self.wk = nn.Linear(d_model, self.h_kv * self.dh, bias=False)
        self.wv = nn.Linear(d_model, self.h_kv * self.dh, bias=False)
        self.wo = nn.Linear(self.h * self.dh, d_model, bias=False)

    def forward(self, x, cos=None, sin=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.dh).transpose(1, 2)       # (B, H, T, dh)
        k = self.wk(x).view(B, T, self.h_kv, self.dh).transpose(1, 2)    # (B, Hkv, T, dh)
        v = self.wv(x).view(B, T, self.h_kv, self.dh).transpose(1, 2)
        if cos is not None:                                               # RoPE on q and k only
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.h // self.h_kv
        if rep > 1:   # query heads [g*rep, (g+1)*rep) share KV head g
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out, _ = sdpa(q, k, v, causal=True)                               # (B, H, T, dh)
        out = out.transpose(1, 2).contiguous().view(B, T, self.h * self.dh)  # concat heads
        return self.wo(out)                                               # mix heads

torch.manual_seed(0)
for hkv in (8, 2, 1):
    m = Attention(64, 8, hkv)
    x = torch.randn(2, 10, 64)
    y = m(x)
    assert y.shape == x.shape
    # reference with F.sdpa and explicit expand
    q = m.wq(x).view(2, 10, 8, 8).transpose(1, 2)
    k = m.wk(x).view(2, 10, hkv, 8).transpose(1, 2).repeat_interleave(8 // hkv, 1)
    v = m.wv(x).view(2, 10, hkv, 8).transpose(1, 2).repeat_interleave(8 // hkv, 1)
    ref = m.wo(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(2, 10, 64))
    assert torch.allclose(y, ref, atol=1e-5)
    nparams = sum(p.numel() for p in m.parameters())
    assert nparams == 64 * 64 * 2 + 2 * 64 * (hkv * 8)
    # causality
    x2 = x.clone(); x2[:, -1] += 3
    assert torch.allclose(m(x2)[:, :-1], y[:, :-1], atol=1e-5)
    cos, sin = rope_cache(10, 8)
    assert m(x, cos, sin).shape == x.shape
print("03 attention ok")
