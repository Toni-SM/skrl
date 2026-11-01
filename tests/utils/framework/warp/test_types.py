import typing

import warp as wp

from skrl.agents.warp.ppo import PPO
from skrl.utils.framework.warp.types import Tensor


def test_tensor_alias():
    assert Tensor is wp.array


def test_annotations_resolve_to_framework_type():
    hints = typing.get_type_hints(PPO.act)
    assert hints["observations"] is wp.array
    assert hints["states"] == wp.array | None
    assert hints["return"] == tuple[wp.array, dict[str, typing.Any]]
