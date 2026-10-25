import torch
import torch.nn.functional as F
# ==== SOLUTION ====
def token_logprobs(logits, input_ids):
    """
    logits: (B, T, V) —— 模型对 input_ids 的输出；位置 t 的 logits 预测 token t+1。
    返回 (B, T-1)：logp[b, t] = log p(input_ids[b, t+1] | input_ids[b, :t+1])
    """
    logits = logits[:, :-1].float()                 # 最后一个位置没有 "下一个 token"
    targets = input_ids[:, 1:]                      # 对齐：右移一位
    logp = F.log_softmax(logits, dim=-1)            # (B, T-1, V)
    return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

def masked_mean(x, mask, dim=None):
    """mask: 与 x 同形状，1 = 计入。"""
    mask = mask.to(x.dtype)
    if dim is None:
        return (x * mask).sum() / mask.sum().clamp(min=1)
    return (x * mask).sum(dim) / mask.sum(dim).clamp(min=1)

def response_mask(prompt_lens, attention_mask):
    """
    prompt_lens: (B,) 每条 prompt 的长度；attention_mask: (B, T)，1 = 真实 token（右侧 padding）。
    返回与 token_logprobs 对齐的 (B, T-1) mask：只有 response token 为 1。
    """
    B, T = attention_mask.shape
    pos = torch.arange(1, T).unsqueeze(0)           # 被预测 token 的位置 1..T-1
    is_resp = pos >= prompt_lens.unsqueeze(1)
    return is_resp & attention_mask[:, 1:].bool()
# ==== TESTS ====
torch.manual_seed(0)
B, T, V = 2, 7, 13
logits = torch.randn(B, T, V); ids = torch.randint(0, V, (B, T))
lp = token_logprobs(logits, ids)
ref = -F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1), reduction="none").view(B, T - 1)
assert torch.allclose(lp, ref, atol=1e-5)
attn = torch.ones(B, T, dtype=torch.long); attn[1, -2:] = 0
m = response_mask(torch.tensor([3, 2]), attn)
expected = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0]], dtype=torch.bool)
assert torch.equal(m, expected)
assert torch.allclose(masked_mean(lp, m), lp[m].mean(), atol=1e-6)
print("ok")
