"""Self-contained test of the CAA forward-hook code published in m11 §6 (needs torch)."""
import torch, torch.nn as nn

# ---- code as published ----
@torch.no_grad()
def mean_diff_vector(model, layer, pos_batches, neg_batches):
    acts = {}
    def grab(_m, _inp, out):
        acts["h"] = (out[0] if isinstance(out, tuple) else out)[:, -1, :]
    handle = model.model.layers[layer].register_forward_hook(grab)
    def mean_act(batches):
        total, n = 0, 0
        for ids in batches:
            model(ids); total = total + acts["h"].sum(0); n += ids.shape[0]
        return total / n
    v = mean_act(pos_batches) - mean_act(neg_batches)
    handle.remove()
    return v

def add_steering(model, layer, v, alpha):
    def hook(_m, _inp, out):
        if isinstance(out, tuple):
            return (out[0] + alpha * v,) + out[1:]
        return out + alpha * v
    return model.model.layers[layer].register_forward_hook(hook)

# ---- tiny HF-like model: layers return tuples like Llama decoder layers ----
class Layer(nn.Module):
    def __init__(s, d): super().__init__(); s.lin = nn.Linear(d, d)
    def forward(s, h): return (h + torch.tanh(s.lin(h)), None)
class Inner(nn.Module):
    def __init__(s, V, d, L): super().__init__(); s.emb = nn.Embedding(V, d); s.layers = nn.ModuleList([Layer(d) for _ in range(L)])
class Model(nn.Module):
    def __init__(s, V=50, d=8, L=3): super().__init__(); s.model = Inner(V, d, L); s.head = nn.Linear(d, V)
    def forward(s, ids):
        h = s.model.emb(ids)
        for l in s.model.layers: h = l(h)[0]
        return s.head(h)

torch.manual_seed(0)
m = Model().eval()
pos = [torch.randint(0, 25, (4, 6)) for _ in range(3)]
neg = [torch.randint(25, 50, (4, 6)) for _ in range(3)]
v = mean_diff_vector(m, 1, pos, neg)
assert v.shape == (8,)
assert len(m.model.layers[1]._forward_hooks) == 0       # hook removed

# manual check of the mean difference
def last_h(ids):
    h = m.model.emb(ids)
    for l in m.model.layers[:2]: h = l(h)[0]
    return h[:, -1, :]
with torch.no_grad():
    ref = torch.cat([last_h(b) for b in pos]).mean(0) - torch.cat([last_h(b) for b in neg]).mean(0)
assert torch.allclose(v, ref, atol=1e-6)

# steering: alpha=0 is a no-op, alpha!=0 changes the output; removal restores
ids = torch.randint(0, 50, (2, 6))
with torch.no_grad():
    base = m(ids)
    h = add_steering(m, 1, v, 0.0); assert torch.allclose(m(ids), base); h.remove()
    h = add_steering(m, 1, v, 4.0); steered = m(ids); h.remove()
    assert not torch.allclose(steered, base)
    assert torch.allclose(m(ids), base)
print("OK")
