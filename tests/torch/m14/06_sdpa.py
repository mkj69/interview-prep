import math
import torch
import torch.nn.functional as F
# ==== SOLUTION ====
def sdpa(q, k, v, causal: bool = False, key_padding_mask=None):
    """
    q: (B, H, Tq, D)   k, v: (B, H, Tk, D)
    key_padding_mask: (B, Tk) bool，True = 真实 token，False = padding
    causal 且 Tq < Tk 时（例如有 KV cache），query i 对应绝对位置 Tk - Tq + i
    返回: (B, H, Tq, D)
    """
    B, H, Tq, D = q.shape
    Tk = k.shape[-2]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)          # (B, H, Tq, Tk)
    allowed = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device)
    if causal:
        # 下三角，对角线向右平移 Tk - Tq
        allowed = torch.tril(allowed, diagonal=Tk - Tq)
    allowed = allowed.view(1, 1, Tq, Tk)
    if key_padding_mask is not None:
        allowed = allowed & key_padding_mask.view(B, 1, 1, Tk)
    scores = scores.masked_fill(~allowed, float("-inf"))        # 在 softmax 之前 mask
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)   # 整行都被 mask 时 softmax 给 nan -> 置 0
    return attn @ v
# ==== TESTS ====
torch.manual_seed(0)
B, H, T, D = 2, 3, 6, 8
q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
assert torch.allclose(sdpa(q, k, v), F.scaled_dot_product_attention(q, k, v), atol=1e-5)
assert torch.allclose(sdpa(q, k, v, causal=True),
                      F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-5)
pad = torch.ones(B, T, dtype=torch.bool); pad[1, -2:] = False       # 第 2 条序列右侧 2 个 pad
m = torch.tril(torch.ones(T, T, dtype=torch.bool)).view(1, 1, T, T) & pad.view(B, 1, 1, T)
ref = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
assert torch.allclose(sdpa(q, k, v, causal=True, key_padding_mask=pad), ref, atol=1e-5)
# Tq < Tk：最后 2 个 query 应等于完整 causal attention 的最后 2 行
full = sdpa(q, k, v, causal=True)
assert torch.allclose(sdpa(q[:, :, -2:], k, v, causal=True), full[:, :, -2:], atol=1e-5)
# 因果性：修改未来的 k/v 不影响过去的输出
k2, v2 = k.clone(), v.clone(); k2[..., -1, :] += 10; v2[..., -1, :] += 10
assert torch.allclose(sdpa(q, k2, v2, causal=True)[:, :, :-1], full[:, :, :-1], atol=1e-5)
print("ok")
