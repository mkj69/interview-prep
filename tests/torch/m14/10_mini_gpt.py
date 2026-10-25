import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# ==== SOLUTION ====
class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h, self.hd = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d)                 # 一次投影出 q, k, v
        self.proj = nn.Linear(d, d)

    def forward(self, x, kv_cache=None):
        """kv_cache: None 或 (k_past, v_past)，各 (B, h, L, hd)。返回 (y, (k, v))。"""
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q, k, v = (t.view(B, T, self.h, self.hd).transpose(1, 2) for t in (q, k, v))
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        Tk = k.shape[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)        # (B, h, T, Tk)
        mask = torch.tril(torch.ones(T, Tk, dtype=torch.bool, device=x.device), diagonal=Tk - T)
        att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), (k, v)

class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, kv_cache=None):
        a, new_cache = self.attn(self.ln1(x), kv_cache)   # pre-norm + 残差
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, new_cache

class MiniGPT(nn.Module):
    def __init__(self, vocab, d=64, n_heads=4, n_layers=2, max_len=128):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)              # 可学习绝对位置
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight               # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):                                # GPT-2 风格小初始化：初始 loss ≈ ln(V)
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, caches=None):
        """idx: (B, T) long。caches: None 或每层一个 (k, v)。返回 logits (B, T, V), new_caches。"""
        B, T = idx.shape
        past = 0 if caches is None else caches[0][0].shape[2]
        assert past + T <= self.max_len
        pos = torch.arange(past, past + T, device=idx.device)   # 有 cache 时位置要接着往后数
        x = self.tok(idx) + self.pos(pos)
        new_caches = []
        for i, blk in enumerate(self.blocks):
            x, c = blk(x, None if caches is None else caches[i])
            new_caches.append(c)
        return self.head(self.ln_f(x)), new_caches

def train_on_string(text: str, steps=300, T=32, B=16, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    model = MiniGPT(len(chars), d=64, n_heads=4, n_layers=2, max_len=T)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses = []
    model.train()
    for step in range(steps):
        ix = torch.randint(0, len(data) - T - 1, (B,))
        x = torch.stack([data[i:i + T] for i in ix])            # (B, T)
        y = torch.stack([data[i + 1:i + T + 1] for i in ix])    # 目标 = 右移一位
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return model, losses, stoi
# ==== TESTS ====
text = "hello world. the quick brown fox jumps over the lazy dog. " * 40
model, losses, stoi = train_on_string(text, steps=300)
V = len(stoi)
first, last = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
print(f"loss {first:.3f} -> {last:.3f} (uniform = {math.log(V):.3f})")
assert abs(first - math.log(V)) < 1.0      # 初始 loss 应接近 ln(V)
assert last < 0.5 * first                  # 明显下降
# 因果性：改变后面的 token 不影响前面的 logits
model.eval()
x = torch.randint(0, V, (1, 10)); x2 = x.clone(); x2[0, 7] = (x[0, 7] + 1) % V
with torch.no_grad():
    assert torch.allclose(model(x)[0][:, :7], model(x2)[0][:, :7], atol=1e-5)
print("ok")
