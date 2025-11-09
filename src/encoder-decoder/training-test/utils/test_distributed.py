import os

import pytest

from training.utils import distributed


def test_world_info_defaults(monkeypatch):
    # Ensure no interfering env vars
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    rank, local_rank, world_size = distributed.world_info()
    assert rank == 0
    assert local_rank == 0
    assert world_size == 1


def test_world_info_reads_env(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")

    rank, local_rank, world_size = distributed.world_info()
    assert rank == 3
    assert local_rank == 1
    assert world_size == 8


def test_init_dist_if_needed_no_init_when_world_size_1(monkeypatch):
    # WORLD_SIZE = 1 → should not init process group
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    # If this is called, test should fail
    def _bad_init_process_group(*args, **kwargs):
        raise AssertionError("init_process_group should not be called when WORLD_SIZE == 1")

    # Likewise, set_device should not be needed
    def _bad_set_device(*args, **kwargs):
        raise AssertionError("set_device should not be called when WORLD_SIZE == 1")

    monkeypatch.setattr(distributed.torch.distributed, "init_process_group", _bad_init_process_group, raising=True)
    monkeypatch.setattr(distributed.torch.cuda, "set_device", _bad_set_device, raising=True)

    rank, local_rank, world_size = distributed.init_dist_if_needed()
    assert rank == 0
    assert local_rank == 0
    assert world_size == 1


def test_init_dist_if_needed_initializes_when_world_size_gt_1(monkeypatch):
    # Simulate multi-process env where dist is not yet initialized
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")

    calls = {"set_device": None, "init_pg": None}

    def _fake_is_initialized():
        return False

    def _fake_set_device(dev):
        calls["set_device"] = dev

    def _fake_init_process_group(backend, init_method):
        calls["init_pg"] = {"backend": backend, "init_method": init_method}

    monkeypatch.setattr(distributed.torch.distributed, "is_initialized", _fake_is_initialized, raising=True)
    monkeypatch.setattr(distributed.torch.cuda, "set_device", _fake_set_device, raising=True)
    monkeypatch.setattr(distributed.torch.distributed, "init_process_group", _fake_init_process_group, raising=True)

    rank, local_rank, world_size = distributed.init_dist_if_needed()

    assert (rank, local_rank, world_size) == (2, 1, 4)
    assert calls["set_device"] == 1
    assert calls["init_pg"] == {"backend": "nccl", "init_method": "env://"}


def test_init_dist_if_needed_skips_when_already_initialized(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")

    def _fake_is_initialized():
        return True

    # If these are called, it's a bug; raise.
    def _bad_set_device(*args, **kwargs):
        raise AssertionError("set_device should not be called when already initialized")

    def _bad_init_process_group(*args, **kwargs):
        raise AssertionError("init_process_group should not be called when already initialized")

    monkeypatch.setattr(distributed.torch.distributed, "is_initialized", _fake_is_initialized, raising=True)
    monkeypatch.setattr(distributed.torch.cuda, "set_device", _bad_set_device, raising=True)
    monkeypatch.setattr(distributed.torch.distributed, "init_process_group", _bad_init_process_group, raising=True)

    rank, local_rank, world_size = distributed.init_dist_if_needed()
    assert (rank, local_rank, world_size) == (1, 0, 4)


def test_is_main_process_true_when_rank_zero(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    assert distributed.is_main_process() is True


def test_is_main_process_false_when_rank_non_zero(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    assert distributed.is_main_process() is False
