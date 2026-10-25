import math
import torch
# ==== SOLUTION ====
class KVCache:
    """预分配 (B, H, T_max, d)，按位置写入；比每步 torch.cat 少分配内存。"""
    def __init__(self, B, H, T_max, d, dtype=torch.float32):
        self.k = torch.zeros(B, H, T_max, d, dtype=dtype)
        self.v = torch.zeros(B, H, T_max, d, dtype=dtype)
        self.len = 0

    def append(self, k_new, v_new):                      # (B, H, t, d)
        t = k_new.shape[2]
        self.k[:, :, self.len:self.len + t] = k_new
        self.v[:, :, self.len:self.len + t] = v_new
        self.len += t
        return self.k[:, :, :self.len], self.v[:, :, :self.len]

def decode_step_attention(q_new, k_new, v_new, cache: KVCache):
    """
    q_new, k_new, v_new: (B, H, 1, d) —— 当前这一个新 token 的投影。
    新 query 可以看见缓存里全部过去 token + 自己，所以不需要 causal mask。
    """
    K, V = cache.append(k_new, v_new)                    # (B, H, L, d)
    d = q_new.shape[-1]
    scores = (q_new @ K.transpose(-2, -1)) / math.sqrt(d)   # (B, H, 1, L)
    return scores.softmax(dim=-1) @ V                       # (B, H, 1, d)
# ==== TESTS ====
torch.manual_seed(0)
B, H, T, d = 2, 3, 9, 8
q, k, v = (torch.randn(B, H, T, d) for _ in range(3))
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
full = ((q @ k.transpose(-2, -1)) / math.sqrt(d)).masked_fill(~mask, float("-inf")).softmax(-1) @ v
cache = KVCache(B, H, 32, d)
outs = [decode_step_attention(q[:, :, t:t+1], k[:, :, t:t+1], v[:, :, t:t+1], cache) for t in range(T)]
assert torch.allclose(torch.cat(outs, dim=2), full, atol=1e-5)
assert cache.len == T
print("ok")
