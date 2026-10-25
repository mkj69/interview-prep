import torch

def sample_logits(logits, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0):
    """logits: (B, V). Returns next token ids (B,)."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]       # (B,1)
        logits = logits.masked_fill(logits < kth, float('-inf'))
    if top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(dim=-1)
        cum = probs.cumsum(dim=-1)
        remove = (cum - probs) >= top_p             # mass before this token already >= p
        sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
        logits = torch.full_like(logits, float('-inf')).scatter(-1, idx, sorted_logits)
    if min_p > 0.0:
        probs = logits.softmax(dim=-1)
        thresh = min_p * probs.max(dim=-1, keepdim=True).values
        logits = logits.masked_fill(probs < thresh, float('-inf'))
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

if __name__ == '__main__':
    torch.manual_seed(0)
    base = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05]).log()
    lg = base.repeat(20000, 1)
    assert (sample_logits(lg[:3], temperature=0) == 0).all()
    s = sample_logits(lg, top_k=2); assert s.max() <= 1
    assert abs((s == 0).float().mean().item() - 0.5 / 0.7) < 0.02
    s = sample_logits(lg, top_p=0.6); assert set(s.unique().tolist()) <= {0, 1}
    s = sample_logits(lg, top_p=0.9); assert set(s.unique().tolist()) == {0, 1, 2, 3}
    s = sample_logits(lg, min_p=0.3); assert set(s.unique().tolist()) == {0, 1, 2}
    print('ok')
