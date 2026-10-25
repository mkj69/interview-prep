import torch
import torch.nn.functional as F

def token_logprobs(logits, input_ids):
    """logits: (B, T, V) from model(input_ids); input_ids: (B, T).
    Returns (B, T-1): log pi(x_{t+1} | x_{<=t})."""
    logits = logits[:, :-1, :].float()          # position t predicts token t+1
    targets = input_ids[:, 1:]                    # shift left
    logp = F.log_softmax(logits, dim=-1)          # (B, T-1, V)
    return torch.gather(logp, -1, targets.unsqueeze(-1)).squeeze(-1)

def masked_mean(x, mask, dim=None):
    mask = mask.to(x.dtype)
    if dim is None:
        return (x * mask).sum() / mask.sum().clamp(min=1)
    return (x * mask).sum(dim) / mask.sum(dim).clamp(min=1)

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, V = 2, 6, 11
    logits = torch.randn(B, T, V); ids = torch.randint(0, V, (B, T))
    lp = token_logprobs(logits, ids)
    ref = -F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1), reduction="none").view(B, T - 1)
    assert torch.allclose(lp, ref, atol=1e-6)
    # response mask: only tokens after a 2-token prompt count
    mask = torch.zeros(B, T - 1); mask[:, 2:] = 1
    assert torch.allclose(masked_mean(lp, mask), lp[:, 2:].mean(), atol=1e-6)
    print("ok")
