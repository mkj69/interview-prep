# DDP minimal example. Runs on CPU with 2 processes (gloo backend) so it can be tested without GPUs.
# On a GPU node you would launch with: torchrun --nproc_per_node=8 train.py  and use backend="nccl".
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29511"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(0)                      # same init on every rank (DDP also broadcasts rank 0's weights)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    ddp = DDP(model)                          # registers autograd hooks -> bucketed all-reduce of grads
    opt = torch.optim.SGD(ddp.parameters(), lr=0.1)

    g = torch.Generator().manual_seed(123)
    X = torch.randn(8, 8, generator=g); y = torch.randn(8, 1, generator=g)
    Xr, yr = X.chunk(world_size)[rank], y.chunk(world_size)[rank]   # each rank: its shard of the batch

    loss = torch.nn.functional.mse_loss(ddp(Xr), yr)
    opt.zero_grad(); loss.backward(); opt.step()   # backward() already averaged grads across ranks

    # reference: single-process model on the full batch must give identical weights
    torch.manual_seed(0)
    ref = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    ropt = torch.optim.SGD(ref.parameters(), lr=0.1)
    rl = torch.nn.functional.mse_loss(ref(X), y)
    ropt.zero_grad(); rl.backward(); ropt.step()
    for p, q in zip(model.parameters(), ref.parameters()):
        assert torch.allclose(p, q, atol=1e-6), "DDP != single-process full batch"
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
    print("DDP OK")
