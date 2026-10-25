import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 数据：字符级 ----------
text = "hello world! this is a tiny char-level language model. " * 200
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
V, T, B = len(chars), 32, 16          # 词表、上下文长度、batch
D, H, L = 64, 4, 2                    # 宽度、头数、层数

def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - T - 1, (B,))
    x = torch.stack([src[i : i + T] for i in ix])            # (B, T)
    y = torch.stack([src[i + 1 : i + T + 1] for i in ix])    # 目标 = 右移一位
    return x.to(device), y.to(device)

# ---------- 模型 ----------
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.register_buffer("mask", torch.tril(torch.ones(T, T, dtype=torch.bool)))

    def forward(self, x):                                    # x: (B, t, D)
        b, t, _ = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)               # 各 (B, t, D)
        q, k, v = (z.view(b, t, H, D // H).transpose(1, 2) for z in (q, k, v))  # (B, H, t, Dh)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D // H)  # (B, H, t, t)
        att = att.masked_fill(~self.mask[:t, :t], float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, D)   # 合并 heads
        return self.proj(y)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn = CausalSelfAttention()
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))                       # Pre-Norm + 残差
        return x + self.mlp(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.blocks = nn.Sequential(*[Block() for _ in range(L)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)

    def forward(self, idx, targets=None):                    # idx: (B, t)
        t = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        logits = self.head(self.ln_f(self.blocks(x)))        # (B, t, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=1.0):
        for _ in range(n_new):
            logits, _ = self(idx[:, -T:])                    # 只保留最近 T 个 token
            probs = (logits[:, -1] / temperature).softmax(-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

# ---------- 训练 ----------
model = TinyGPT().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")
_, loss0 = model(*get_batch("train"))
print(f"init loss {loss0.item():.3f}  (ln V = {math.log(V):.3f})")
for step in range(500):
    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 100 == 0:
        model.eval()
        with torch.no_grad():
            _, vl = model(*get_batch("val"))
        model.train()
        print(f"step {step:4d}  train {loss.item():.3f}  val {vl.item():.3f}")
model.eval()
start = torch.tensor([[stoi["h"]]], device=device)
print("".join(itos[i] for i in model.generate(start, 60)[0].tolist()))
assert loss.item() < 0.5 * math.log(V)
