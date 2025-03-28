from typing import Union

import hypothesis
import hypothesis.strategies as st
import pytest

import os

import torch

from skrl import _Config, config


@pytest.mark.parametrize("device", [None, "cpu", "cuda", "cuda:0", "cuda:10", "edge-case"])
@pytest.mark.parametrize("validate", [True, False])
def test_parse_device(capsys, device: Union[str, None], validate: bool):
    target_device = None
    if device in [None, "edge-case"]:
        target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif device.startswith("cuda"):
        if validate and int(f"{device}:0".split(":")[1]) >= torch.cuda.device_count():
            target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not target_device:
        target_device = torch.device(device)

    runtime_device = config.torch.parse_device(device, validate=validate)
    assert runtime_device == target_device


@pytest.mark.parametrize("device", [None, "cpu", "cuda", "cuda:0", "cuda:10", "edge-case"])
def test_device(capsys, device: Union[str, None]):
    if device in [None, "edge-case"]:
        target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    # check setter/getter
    config.torch.device = device
    assert config.torch.device == target_device


@hypothesis.given(
    local_rank=st.integers(),
    rank=st.integers(),
    world_size=st.integers(),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
def test_distributed(capsys, local_rank: int, rank: int, world_size: int):
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    is_distributed = world_size > 1

    if is_distributed:
        with pytest.raises(ValueError, match="Error initializing torch.distributed"):
            config = _Config()
        return
    else:
        config = _Config()
    assert config.torch.local_rank == local_rank
    assert config.torch.rank == rank
    assert config.torch.world_size == world_size
    assert config.torch.is_distributed == is_distributed
    assert config.torch._device == f"cuda:{local_rank}"
