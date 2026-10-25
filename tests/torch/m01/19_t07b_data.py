import torch
from torch.utils.data import Dataset, DataLoader

class SeqDataset(Dataset):
    def __init__(self, seqs):                # seqs: 变长的 token id 列表
        self.seqs = seqs
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, i):                # 返回单个样本
        return torch.tensor(self.seqs[i], dtype=torch.long)

PAD = 0
def collate(batch):                          # batch: list of 1-D tensors，长度各不相同
    lengths = torch.tensor([len(s) for s in batch])
    L = int(lengths.max())
    ids = torch.full((len(batch), L), PAD, dtype=torch.long)
    for i, s in enumerate(batch):
        ids[i, : len(s)] = s                 # 右 padding
    mask = torch.arange(L)[None, :] < lengths[:, None]   # (B, L) True = 真实 token
    return ids, mask
# 等价工具：torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=PAD)

ds = SeqDataset([[5, 6, 7], [8, 9], [1, 2, 3, 4], [3]])
dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)
ids, mask = next(iter(dl))
print(ids)    # tensor([[5, 6, 7], [8, 9, 0]])
print(mask)   # tensor([[ True,  True,  True], [ True,  True, False]])
