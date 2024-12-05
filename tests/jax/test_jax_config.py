from typing import Union

import pytest

import jax

from skrl import config


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
