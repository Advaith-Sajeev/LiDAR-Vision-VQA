"""Distributed training utilities"""

import os
import torch


def world_info():
    """Get distributed training info."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def init_dist_if_needed():
    """Initialize distributed training if world_size > 1."""
    rank, local_rank, world_size = world_info()
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size


def is_main_process() -> bool:
    """Check if this is the main process."""
    return int(os.environ.get("RANK", "0")) == 0
