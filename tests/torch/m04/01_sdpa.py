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

torch.manual_seed(0)
B, H, T, dh = 2, 4, 7, 16
q, k, v = (torch.randn(B, H, T, dh) for _ in range(3))
out, attn = sdpa(q, k, v, causal=True)
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
assert torch.allclose(out, ref, atol=1e-5)
assert torch.allclose(attn.sum(-1), torch.ones(B, H, T))
assert torch.all(attn.triu(1) == 0)
# decoding with cache: last query vs all keys == last row of full causal
o1, _ = sdpa(q[:, :, -1:], k, v, causal=True)
assert torch.allclose(o1, out[:, :, -1:], atol=1e-6)
# padding mask
pad = torch.ones(B, T, dtype=torch.bool); pad[1, -2:] = False
o2, a2 = sdpa(q, k, v, causal=False, key_padding_mask=pad)
ref2 = F.scaled_dot_product_attention(q, k, v, attn_mask=pad[:, None, None, :])
assert torch.allclose(o2, ref2, atol=1e-5) and torch.all(a2[1, :, :, -2:] == 0)
print("01 sdpa ok")
