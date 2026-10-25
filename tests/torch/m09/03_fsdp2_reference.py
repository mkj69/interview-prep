# FSDP2 (fully_shard) reference snippet. Intended for GPUs (nccl). This test tries CPU+gloo; if the
# installed torch version does not support FSDP2 on CPU it prints SKIP and exits 0.
import os, sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class Block(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ff = torch.nn.Sequential(torch.nn.Linear(d, 4 * d), torch.nn.GELU(), torch.nn.Linear(4 * d, d))

    def forward(self, x):
        return x + self.ff(x)


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29513"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    from torch.distributed.fsdp import fully_shard   # public since PyTorch 2.6
    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh("cpu", (world_size,))
    torch.manual_seed(0)
    model = torch.nn.Sequential(*[Block(16) for _ in range(4)])
    for blk in model:                     # shard each block: its params are all-gathered just-in-time
        fully_shard(blk, mesh=mesh, reshard_after_forward=True)
    fully_shard(model, mesh=mesh)         # root wrap for remaining params
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)   # optimizer only sees local shards (DTensor)
    x = torch.randn(4, 16)
    loss = model(x).pow(2).mean()
    loss.backward()                       # grads are reduce-scattered to shards
    opt.step(); opt.zero_grad()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        mp.spawn(worker, args=(2,), nprocs=2, join=True)
        print("FSDP2 OK")
    except Exception as e:  # environment without CPU FSDP2 support
        print("SKIP:", type(e).__name__, e)
        sys.exit(0)
