import typing

import torch

from skrl.agents.torch.ppo import PPO
from skrl.utils.framework.torch.types import Tensor


def test_tensor_alias():
    assert Tensor is torch.Tensor


def test_annotations_resolve_to_framework_type():
    hints = typing.get_type_hints(PPO.act)
    assert hints["observations"] is torch.Tensor
    assert hints["states"] == torch.Tensor | None
    assert hints["return"] == tuple[torch.Tensor, dict[str, typing.Any]]
