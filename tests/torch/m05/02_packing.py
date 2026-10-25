import torch

def packed_attention_mask(doc_ids):
    T = doc_ids.size(0)
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    same_doc = doc_ids[:, None] == doc_ids[None, :]
    return causal & same_doc

def packed_position_ids(doc_ids):
    T = doc_ids.size(0)
    idx = torch.arange(T)
    is_start = torch.ones(T, dtype=torch.bool)
    is_start[1:] = doc_ids[1:] != doc_ids[:-1]
    seg_start = torch.cummax(torch.where(is_start, idx, torch.zeros_like(idx)), dim=0).values
    return idx - seg_start

def boundary_loss_mask(doc_ids):
    m = torch.zeros_like(doc_ids, dtype=torch.bool)
    m[:-1] = doc_ids[1:] == doc_ids[:-1]
    return m

d = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
M = packed_attention_mask(d)
assert M[4, 3] and not M[3, 2] and not M[5, 4] and M[7, 5] and not M[0, 1]
assert packed_position_ids(d).tolist() == [0, 1, 2, 0, 1, 0, 1, 2]
assert boundary_loss_mask(d).tolist() == [True, True, False, True, False, True, True, False]
print("ok")
