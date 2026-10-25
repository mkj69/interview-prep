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
class Block(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads=None):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = Attention(d, n_heads, n_kv_heads)
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)    # Pre-Norm: normalize only the branch input
        x = x + self.mlp(self.norm2(x))               # residual stream x itself is never normalized
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab, d=128, n_layers=4, n_heads=4, n_kv_heads=2, max_T=256, tie=True):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, n_kv_heads) for _ in range(n_layers)])
        self.norm = RMSNorm(d)                        # final norm before the LM head
        self.head = nn.Linear(d, vocab, bias=False)
        if tie:
            self.head.weight = self.tok.weight        # weight tying: one (V, d) matrix
        cos, sin = rope_cache(max_T, d // n_heads)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx)                             # (B, T, d)
        cos, sin = self.cos[:T], self.sin[:T]
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = self.head(self.norm(x))              # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

torch.manual_seed(0)
text = "hello transformer world! " * 40
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
model = TinyGPT(len(chars), d=64, n_layers=2, n_heads=4, n_kv_heads=2, max_T=64)
assert model.head.weight is model.tok.weight
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
T = 32
first = None
for step in range(150):
    ix = torch.randint(0, len(data) - T - 1, (16,))
    xb = torch.stack([data[i:i + T] for i in ix]); yb = torch.stack([data[i + 1:i + T + 1] for i in ix])
    _, loss = model(xb, yb)
    opt.zero_grad(); loss.backward(); opt.step()
    first = first or loss.item()
print("loss", first, "->", loss.item())
assert loss.item() < 0.5 * first
# causality of the full model
model.eval()
x = data[:20][None]
l1, _ = model(x); x2 = x.clone(); x2[0, -1] = (x2[0, -1] + 1) % len(chars)
l2, _ = model(x2)
assert torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-5)
print("05 tiny gpt ok")
