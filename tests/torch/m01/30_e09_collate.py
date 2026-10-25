import torch

def lm_collate(batch, pad_id=0):
    """batch: list of (prompt_ids, response_ids)，都是 list[int]。
    返回 input_ids (B, L), labels (B, L), attention_mask (B, L)；
    labels 中 prompt 和 padding 位置为 -100，只在 response 上算 loss。"""
    seqs = [p + r for p, r in batch]
    L = max(len(s) for s in seqs)
    B = len(seqs)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    labels = torch.full((B, L), -100, dtype=torch.long)
    attn = torch.zeros(B, L, dtype=torch.bool)
    for i, ((p, r), s) in enumerate(zip(batch, seqs)):
        input_ids[i, : len(s)] = torch.tensor(s)
        labels[i, len(p) : len(s)] = torch.tensor(r)
        attn[i, : len(s)] = True
    return input_ids, labels, attn
# 训练时：logits = model(input_ids)；loss 用 logits[:, :-1] 预测 labels[:, 1:]（shift 一位）

ids, labels, attn = lm_collate([([5, 6], [7, 8, 9]), ([1], [2])])
assert ids.tolist() == [[5, 6, 7, 8, 9], [1, 2, 0, 0, 0]]
assert labels.tolist() == [[-100, -100, 7, 8, 9], [-100, 2, -100, -100, -100]]
assert attn.sum().item() == 7
