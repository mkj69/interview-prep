import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# ==== SOLUTION ====
class MultiHeadAttention(nn.Module):
    """n_kv_heads == n_heads: 标准 MHA；n_kv_heads == 1: MQA；介于之间: GQA。"""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int = None):
        super().__init__()
        n_kv_heads = n_kv_heads or n_heads
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.h, self.kvh = n_heads, n_kv_heads
        self.d = d_model // n_heads                       # head_dim
        self.q_proj = nn.Linear(d_model, n_heads * self.d, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d, d_model, bias=False)

    def forward(self, x, causal: bool = True):
        B, T, C = x.shape
        # (B, T, h*d) -> (B, T, h, d) -> (B, h, T, d)
        q = self.q_proj(x).view(B, T, self.h, self.d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kvh, self.d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kvh, self.d).transpose(1, 2)
        if self.kvh != self.h:                            # GQA: 每个 KV head 被 h/kvh 个 Q head 共享
            rep = self.h // self.kvh
            k = k.repeat_interleave(rep, dim=1)           # (B, h, T, d)
            v = v.repeat_interleave(rep, dim=1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)          # (B, h, T, T)
        if causal:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            att = att.masked_fill(~mask, float("-inf"))
        att = att.softmax(dim=-1)
        y = att @ v                                                  # (B, h, T, d)
        y = y.transpose(1, 2).contiguous().view(B, T, self.h * self.d)  # 先 transpose 回去再 view
        return self.o_proj(y)
# ==== TESTS ====
torch.manual_seed(0)
B, T, C, H = 2, 7, 32, 4
x = torch.randn(B, T, C)
mha = MultiHeadAttention(C, H)
ref = nn.MultiheadAttention(C, H, bias=False, batch_first=True)
with torch.no_grad():
    ref.in_proj_weight.copy_(torch.cat([mha.q_proj.weight, mha.k_proj.weight, mha.v_proj.weight], 0))
    ref.out_proj.weight.copy_(mha.o_proj.weight)
causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)      # nn.MHA: True = 屏蔽
y_ref, _ = ref(x, x, x, attn_mask=causal_mask, need_weights=False)
assert torch.allclose(mha(x), y_ref, atol=1e-5)
# GQA: 和 "手动把 KV 复制成 MHA" 的结果一致
gqa = MultiHeadAttention(C, H, n_kv_heads=2)
assert gqa.k_proj.weight.shape == (2 * 8, C)
mha2 = MultiHeadAttention(C, H)
with torch.no_grad():
    mha2.q_proj.weight.copy_(gqa.q_proj.weight); mha2.o_proj.weight.copy_(gqa.o_proj.weight)
    d = 8
    kw = gqa.k_proj.weight.view(2, d, C).repeat_interleave(2, 0).reshape(H * d, C)
    vw = gqa.v_proj.weight.view(2, d, C).repeat_interleave(2, 0).reshape(H * d, C)
    mha2.k_proj.weight.copy_(kw); mha2.v_proj.weight.copy_(vw)
assert torch.allclose(gqa(x), mha2(x), atol=1e-5)
print("ok")
