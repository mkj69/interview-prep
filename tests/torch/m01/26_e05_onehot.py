import torch
import torch.nn.functional as F

def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    out = torch.zeros(*labels.shape, num_classes, device=labels.device)
    return out.scatter_(-1, labels.unsqueeze(-1), 1.0)     # out[..., labels[...]] = 1

labels = torch.tensor([[0, 3], [2, 1]])                     # 任意 batch 形状都可以
assert torch.equal(one_hot(labels, 4), F.one_hot(labels, 4).float())
