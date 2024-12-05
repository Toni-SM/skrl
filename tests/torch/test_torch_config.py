from typing import Union

import pytest

import torch

from skrl import config


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
