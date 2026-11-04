import typing

import warp as wp

from skrl.agents.warp.ppo import PPO
from skrl.utils.framework.warp.types import Device, Tensor


def test_device_alias():
    assert Device == str | wp.Device | None


def test_tensor_alias():
    assert Tensor is wp.array


def test_annotations_resolve_to_framework_type():
    hints = typing.get_type_hints(PPO.act)
    assert hints["observations"] is wp.array
    assert hints["states"] == wp.array | None
    assert hints["return"] == tuple[wp.array, dict[str, typing.Any]]


def test_device_annotations_resolve_to_framework_type():
    hints = typing.get_type_hints(PPO.__init__)
    assert hints["device"] == str | wp.Device | None
