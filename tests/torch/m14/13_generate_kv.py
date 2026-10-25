import os
import torch
# 复用 10_mini_gpt.py 里的 MiniGPT（只执行类定义部分，不跑它的训练测试）
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "10_mini_gpt.py")
ns = {}; exec(open(_path).read().split("# ==== TESTS ====")[0], ns); MiniGPT = ns["MiniGPT"]
# ==== SOLUTION ====
@torch.no_grad()
def generate_no_cache(model, idx, n_new):
    logits_all = []
    for _ in range(n_new):
        logits, _ = model(idx)                       # 每步重算整个前缀：O(T^2) 次 token 计算
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        logits_all.append(logits[:, -1])
        idx = torch.cat([idx, nxt], dim=1)
    return idx, torch.stack(logits_all, dim=1)

@torch.no_grad()
def generate_kv(model, idx, n_new):
    logits, caches = model(idx)                      # prefill: 一次处理整个 prompt
    logits_all = []
    for i in range(n_new):
        last = logits[:, -1]
        logits_all.append(last)
        nxt = last.argmax(-1, keepdim=True)          # (B, 1)
        idx = torch.cat([idx, nxt], dim=1)
        if i < n_new - 1:                            # 最后一个 token 不需要再前向
            logits, caches = model(nxt, caches)      # decode: 只喂 1 个新 token
    return idx, torch.stack(logits_all, dim=1)
# ==== TESTS ====
torch.manual_seed(0)
model = MiniGPT(vocab=50, d=32, n_heads=4, n_layers=3, max_len=64).eval()
prompt = torch.randint(0, 50, (2, 5))
a_idx, a_log = generate_no_cache(model, prompt, 20)
b_idx, b_log = generate_kv(model, prompt, 20)
assert torch.equal(a_idx, b_idx)
assert torch.allclose(a_log, b_log, atol=1e-4), (a_log - b_log).abs().max()
print("ok")
