import torch
import torch.nn.functional as F
# ==== SOLUTION ====
def cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                  ignore_index: int = -100) -> torch.Tensor:
    """logits: (N, C) float; targets: (N,) long. 返回被保留位置的平均 NLL。"""
    logits = logits.float()                                   # bf16 下也在 fp32 里算
    m = logits.max(dim=-1, keepdim=True).values
    lse = m.squeeze(-1) + torch.log(torch.exp(logits - m).sum(dim=-1))   # (N,)
    valid = targets != ignore_index                           # (N,) bool
    safe_t = targets.masked_fill(~valid, 0)                   # -100 不能直接拿去 gather
    tgt_logit = logits.gather(1, safe_t.unsqueeze(1)).squeeze(1)          # (N,)
    nll = (lse - tgt_logit) * valid                           # 被忽略的位置贡献 0
    return nll.sum() / valid.sum().clamp(min=1)               # 除以有效 token 数，不是 N

def lm_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100):
    """语言模型版：logits (B, T, V)，labels (B, T)。"""
    B, T, V = logits.shape
    return cross_entropy(logits.reshape(B * T, V), labels.reshape(B * T), ignore_index)
# ==== TESTS ====
torch.manual_seed(0)
logits = torch.randn(10, 5) * 3
t = torch.randint(0, 5, (10,))
assert torch.allclose(cross_entropy(logits, t), F.cross_entropy(logits, t), atol=1e-5)
t2 = t.clone(); t2[::3] = -100
assert torch.allclose(cross_entropy(logits, t2), F.cross_entropy(logits, t2, ignore_index=-100), atol=1e-5)
lg = torch.randn(2, 6, 11); lb = torch.randint(0, 11, (2, 6)); lb[0, :2] = -100
ref = F.cross_entropy(lg.reshape(-1, 11), lb.reshape(-1), ignore_index=-100)
assert torch.allclose(lm_loss(lg, lb), ref, atol=1e-5)
assert cross_entropy(logits, torch.full((10,), -100)).item() == 0.0   # 全被忽略：不 nan
print("ok")
