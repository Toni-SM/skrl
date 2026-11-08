from __future__ import annotations

from typing import Any

import gymnasium

import torch
import warp as wp

# MuJoCo Playground runs on PyTorch: reuse its PyTorch wrapper and only convert inputs/outputs (avoid duplicate logic)
from skrl.envs.wrappers.torch.playground_envs import PlaygroundWrapper as _TorchPlaygroundWrapper
from skrl.envs.wrappers.warp.base import Wrapper
from skrl.utils.framework.warp.types import Tensor


def _torch2warp(tensor: torch.Tensor | None) -> Tensor | None:
    return None if tensor is None else wp.from_torch(tensor.contiguous())


class PlaygroundWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """MuJoCo Playground environment wrapper.

        See :py:class:`~skrl.envs.wrappers.torch.playground_envs.PlaygroundWrapper` for more details.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._wrapper = _TorchPlaygroundWrapper(env)
        self._device = wp.device_from_torch(self._wrapper.device)
        self._observations = None
        self._states = None

    @property
    def state_space(self) -> gymnasium.Space | None:
        """State space."""
        return self._wrapper.state_space

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space."""
        return self._wrapper.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space."""
        return self._wrapper.action_space

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._wrapper.num_envs

    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        observations, reward, terminated, truncated, info = self._wrapper.step(
            wp.to_torch(actions).to(self._wrapper.device)
        )
        self._observations = _torch2warp(observations)
        self._states = _torch2warp(self._wrapper.state())
        return (
            self._observations,
            _torch2warp(reward),
            _torch2warp(terminated.to(dtype=torch.int8)),
            _torch2warp(truncated.to(dtype=torch.int8)),
            info,
        )

    def state(self) -> Tensor | None:
        """Get the environment state.

        :return: State.
        """
        return self._states

    def reset(self) -> tuple[Tensor, Any]:
        """Reset the environment.

        :return: Observation, info.
        """
        observations, info = self._wrapper.reset()
        self._observations = _torch2warp(observations)
        self._states = _torch2warp(self._wrapper.state())
        return self._observations, info

    def render(self, *args, **kwargs) -> Any:
        """Render the environment."""
        return self._wrapper.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment."""
        self._wrapper.close()
