# all-reduce == reduce-scatter + all-gather, checked with real torch.distributed collectives on CPU (gloo).
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29512"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    x = torch.arange(8, dtype=torch.float32) + 10 * rank      # each rank holds a different vector

    ar = x.clone()
    dist.all_reduce(ar, op=dist.ReduceOp.SUM)                  # every rank gets the sum

    # reduce-scatter (via all_to_all + local sum, since gloo lacks reduce_scatter_tensor)
    chunks = list(x.clone().chunk(world_size))
    recv = [torch.empty_like(c) for c in chunks]
    dist.all_to_all(recv, chunks)                              # recv[j] = rank j's chunk number `rank`
    my_shard = torch.stack(recv).sum(0)                        # = sum over ranks of chunk `rank`

    gathered = [torch.empty_like(my_shard) for _ in range(world_size)]
    dist.all_gather(gathered, my_shard)                        # all-gather the reduced shards
    assert torch.equal(torch.cat(gathered), ar)
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
    print("collectives OK")
