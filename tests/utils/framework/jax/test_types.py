import typing

import jax

from skrl.agents.jax.ppo import PPO
from skrl.utils.framework.jax.types import Device, Tensor


def test_device_alias():
    assert Device == str | jax.Device | None


def test_tensor_alias():
    assert Tensor is jax.Array


def test_annotations_resolve_to_framework_type():
    hints = typing.get_type_hints(PPO.act)
    assert hints["observations"] is jax.Array
    assert hints["states"] == jax.Array | None
    assert hints["return"] == tuple[jax.Array, dict[str, typing.Any]]


def test_device_annotations_resolve_to_framework_type():
    hints = typing.get_type_hints(PPO.__init__)
    assert hints["device"] == str | jax.Device | None
