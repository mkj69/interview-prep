import torch, torch.nn as nn, torch.nn.functional as F

class CachedSelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, cache=None):
        # x: (B, T, d) new tokens; cache: dict with 'k','v' (B, H, S, dh) or None
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split(self.h * self.dh, dim=-1)
        q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        if cache is not None and 'k' in cache:
            k = torch.cat([cache['k'], k], dim=2)
            v = torch.cat([cache['v'], v], dim=2)
        if cache is not None:
            cache['k'], cache['v'] = k, v
        S = k.shape[2]
        # query i (absolute pos S-T+i) may attend keys 0..S-T+i
        mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
        att = (q @ k.transpose(-2, -1)) / self.dh ** 0.5
        att = att.masked_fill(~mask, float('-inf')).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, -1)
        return self.o(y)

class TinyLM(nn.Module):
    def __init__(self, V=50, d=32, H=4, L=2, max_len=64):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.pos = nn.Embedding(max_len, d)
        self.attn = nn.ModuleList(CachedSelfAttention(d, H) for _ in range(L))
        self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d)) for _ in range(L))
        self.ln = nn.ModuleList(nn.LayerNorm(d) for _ in range(2 * L)); self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)

    def forward(self, ids, start=0, caches=None):
        T = ids.shape[1]
        x = self.emb(ids) + self.pos(torch.arange(start, start + T, device=ids.device))
        for i, (a, m) in enumerate(zip(self.attn, self.mlp)):
            x = x + a(self.ln[2*i](x), None if caches is None else caches[i])
            x = x + m(self.ln[2*i+1](x))
        return self.head(self.lnf(x))

@torch.no_grad()
def generate(model, prompt, n_new, use_cache=True):
    ids = prompt
    if not use_cache:
        for _ in range(n_new):
            nxt = model(ids)[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
        return ids
    caches = [dict() for _ in model.attn]
    logits = model(ids, 0, caches)                   # prefill
    for _ in range(n_new):
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        logits = model(nxt, ids.shape[1] - 1, caches)   # decode one token
    return ids

if __name__ == '__main__':
    torch.manual_seed(0)
    m = TinyLM().eval()
    p = torch.randint(0, 50, (3, 7))
    a = generate(m, p, 20, use_cache=False); b = generate(m, p, 20, use_cache=True)
    assert torch.equal(a, b)
    print('ok')
