from typing import Union

import hypothesis
import hypothesis.strategies as st
import pytest

import os

import jax
import jax.numpy as jnp
import numpy as np

from skrl import _Config, config


@pytest.mark.parametrize("device", [None, "cpu", "cuda", "cuda:0", "cuda:10", "edge-case"])
def test_parse_device(capsys, device: Union[str, None]):
    target_device = None
    if device in [None, "edge-case"]:
        target_device = jax.devices()[0]
    elif device.startswith("cuda"):
        try:
            index = int(f"{device}:0".split(":")[1])
            target_device = jax.devices("cuda")[index]
        except Exception as e:
            target_device = jax.devices()[0]
    if not target_device:
        target_device = jax.devices(device)[0]

    runtime_device = config.jax.parse_device(device)
    assert runtime_device == target_device


@pytest.mark.parametrize("device", [None, "cpu", "cuda", "cuda:0", "cuda:10", "edge-case"])
def test_device(capsys, device: Union[str, None]):
    target_device = None
    if device in [None, "edge-case"]:
        target_device = jax.devices()[0]
    elif device.startswith("cuda"):
        try:
            index = int(f"{device}:0".split(":")[1])
            target_device = jax.devices("cuda")[index]
        except Exception as e:
            target_device = jax.devices()[0]
    if not target_device:
        target_device = jax.devices(device)[0]

    # check setter/getter
    config.jax.device = device
    assert config.jax.device == target_device


@hypothesis.given(
    local_rank=st.integers(),
    rank=st.integers(),
    world_size=st.integers(min_value=-1, max_value=1),  # world_size > 1 initializes distributed run
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
def test_distributed(capsys, local_rank: int, rank: int, world_size: int):
    os.environ["JAX_LOCAL_RANK"] = str(local_rank)
    os.environ["JAX_RANK"] = str(rank)
    os.environ["JAX_WORLD_SIZE"] = str(world_size)
    is_distributed = world_size > 1

    config = _Config()
    assert config.jax.local_rank == local_rank
    assert config.jax.rank == rank
    assert config.jax.world_size == world_size
    assert config.jax.is_distributed == is_distributed
    assert config.jax._device == f"cuda:{local_rank}"


@hypothesis.given(key0=st.integers(min_value=0, max_value=2**32), key1=st.integers(min_value=0, max_value=2**32))
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_key(capsys, device: str, key0: int, key1: int):
    config.jax.device = device
    assert isinstance(config.jax.key, jax.Array)
    assert config.jax.key.device == config.jax.device

    # integer
    config.jax.key = key0
    assert isinstance(config.jax.key, jax.Array)
    assert config.jax.key.device == config.jax.device
    assert (config.jax.key == jnp.array([0, key0], dtype=jnp.uint32, device=config.jax.device)).all()

    # NumPy array
    config.jax.key = np.array([key0, key1], dtype=np.uint32)
    assert isinstance(config.jax.key, jax.Array)
    assert config.jax.key.device == config.jax.device
    assert (config.jax.key == jnp.array([0, key1], dtype=jnp.uint32, device=config.jax.device)).all()

    # JAX array
    config.jax.key = jnp.array([key0, key1], dtype=jnp.uint32, device=config.jax.device)
    assert isinstance(config.jax.key, jax.Array)
    assert config.jax.key.device == config.jax.device
    assert (config.jax.key == jnp.array([key0, key1], dtype=jnp.uint32, device=config.jax.device)).all()
