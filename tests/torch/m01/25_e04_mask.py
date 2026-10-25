import torch

def build_attn_mask(pad_mask: torch.Tensor) -> torch.Tensor:
    """pad_mask: (B, T) bool, True = 真实 token（右 padding）。
    返回 (B, 1, T, T) bool, True = 允许 query i 看 key j。"""
    B, T = pad_mask.shape
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=pad_mask.device))  # (T, T)
    key_ok = pad_mask[:, None, None, :]          # (B, 1, 1, T)：只屏蔽 key 方向
    return causal[None, None] & key_ok           # 广播成 (B, 1, T, T)

def masked_attention(q, k, v, allowed):
    """q, k, v: (B, H, T, D); allowed: (B, 1, T, T)"""
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5     # (B, H, T, T)
    scores = scores.masked_fill(~allowed, float("-inf"))
    return scores.softmax(-1) @ v

torch.manual_seed(0)
B, H, T, D = 2, 3, 5, 4
pad = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
out = masked_attention(q, k, v, build_attn_mask(pad))
assert not torch.isnan(out).any()   # 右 padding + causal：每行至少能看到第 0 个 token，不会出现全 -inf 行
# 对照：把第二条序列截成前 3 个 token 单独算，结果应该一致
out_short = masked_attention(q[1:, :, :3], k[1:, :, :3], v[1:, :, :3], build_attn_mask(pad[1:, :3]))
assert torch.allclose(out[1:, :, :3], out_short, atol=1e-6)
