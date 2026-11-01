import typing

import jax

from skrl.agents.jax.ppo import PPO
from skrl.utils.framework.jax.types import Tensor


def test_tensor_alias():
    assert Tensor is jax.Array


def test_annotations_resolve_to_framework_type():
    hints = typing.get_type_hints(PPO.act)
    assert hints["observations"] is jax.Array
    assert hints["states"] == jax.Array | None
    assert hints["return"] == tuple[jax.Array, dict[str, typing.Any]]
