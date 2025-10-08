from typing import Union

import hypothesis
import hypothesis.strategies as st
import pytest

import os

import warp as wp

from skrl import _Config, config


@pytest.mark.parametrize("device", [None, "cpu", "cuda", "cuda:0", "cuda:10", "edge-case"])
def test_parse_device(capsys, device: Union[str, None]):
    target_device = None
    if device in [None, "edge-case"]:
        target_device = wp.get_device()
    elif device.startswith("cuda"):
        try:
            index = int(f"{device}:0".split(":")[1])
            target_device = wp.get_device(f"cuda:{index}")
        except Exception as e:
            target_device = wp.get_device()
    if not target_device:
        target_device = wp.get_device(device)

    runtime_device = config.warp.parse_device(device)
    assert runtime_device == target_device


@pytest.mark.parametrize("device", [None, "cpu", "cuda", "cuda:0", "cuda:10", "edge-case"])
def test_device(capsys, device: Union[str, None]):
    target_device = None
    if device in [None, "edge-case"]:
        target_device = wp.get_device()
    elif device.startswith("cuda"):
        try:
            index = int(f"{device}:0".split(":")[1])
            target_device = wp.get_device(f"cuda:{index}")
        except Exception as e:
            target_device = wp.get_device()
    if not target_device:
        target_device = wp.get_device(device)

    # check setter/getter
    config.warp.device = device
    assert config.warp.device == target_device


@hypothesis.given(key0=st.integers(min_value=0, max_value=2**32), key1=st.integers(min_value=0, max_value=2**32))
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_key(capsys, device: str, key0: int, key1: int):
    config.warp.device = device
    assert isinstance(config.warp.key, int)

    # integer
    config.warp.key = key0
    assert isinstance(config.warp.key, int)
    assert config.warp.key == key0
