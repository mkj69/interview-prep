import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 0.5, -1.0]])      # (N=1, C=3)，未经 softmax
target = torch.tensor([0])
ce = F.cross_entropy(logits, target)
manual = -(logits[0, 0] - torch.logsumexp(logits[0], dim=0))   # -log softmax(z)_y
assert torch.allclose(ce, manual)
wrong = F.cross_entropy(logits.softmax(-1), target)  # 错：把概率当 logits，又 softmax 了一次
print(round(ce.item(), 4), round(wrong.item(), 4))   # 0.2413 0.7017

# 语言模型：(B, T, V) 的 logits，(B, T) 的目标，padding 位置标 -100
B, T, V = 2, 4, 10
logits = torch.randn(B, T, V)
targets = torch.tensor([[1, 2, -100, -100],
                        [3, -100, -100, -100]])
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))   # ignore_index 默认就是 -100
keep = targets.view(-1) != -100
assert torch.allclose(loss, F.cross_entropy(logits.view(-1, V)[keep], targets.view(-1)[keep]))
# 不 flatten 也行，但类别维必须在 dim=1 → (B, V, T)
assert torch.allclose(loss, F.cross_entropy(logits.transpose(1, 2), targets))
