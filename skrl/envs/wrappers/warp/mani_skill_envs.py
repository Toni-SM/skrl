from __future__ import annotations

from typing import Any

import gymnasium

import torch
import warp as wp

from skrl import config

# ManiSkill runs on PyTorch: reuse its PyTorch wrapper and only convert inputs/outputs (avoid duplicate logic)
from skrl.envs.wrappers.torch.mani_skill_envs import ManiSkillWrapper as _TorchManiSkillWrapper
from skrl.envs.wrappers.warp.base import Wrapper
from skrl.utils.framework.warp.types import Tensor


def _torch2warp(tensor: torch.Tensor | None) -> Tensor | None:
    return None if tensor is None else wp.from_torch(tensor)


def _warp2torch(array: Tensor | None) -> torch.Tensor | None:
    return None if array is None else wp.to_torch(array)


class ManiSkillWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """ManiSkill environment wrapper.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._wrapper = _TorchManiSkillWrapper(env)
        self._wrapper._seed = config.warp.key
        self._observations = None
        self._states = None
        self._final_observations = None
        self._final_states = None

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

    def _convert(
        self,
        observations: torch.Tensor,
        states: torch.Tensor | None,
        final_observations: torch.Tensor,
        final_states: torch.Tensor | None,
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor | None]:
        """Convert the PyTorch wrapper's observations/states and final observations/states.

        :param observations: Observations returned by the PyTorch wrapper's :py:meth:`step`/:py:meth:`reset`.
        :param states: States returned by the PyTorch wrapper's :py:meth:`state`.
        :param final_observations: Final observations returned by the PyTorch wrapper's :py:meth:`final`.
        :param final_states: Final states returned by the PyTorch wrapper's :py:meth:`final`.

        :return: Observation, state, final observation, final state.
        """
        converted_observations = _torch2warp(observations)
        converted_states = _torch2warp(states)
        # avoid converting the same tensor twice when no episode ended
        return (
            converted_observations,
            converted_states,
            converted_observations if final_observations is observations else _torch2warp(final_observations),
            converted_states if final_states is states else _torch2warp(final_states),
        )

    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        observations, reward, terminated, truncated, info = self._wrapper.step(_warp2torch(actions))
        self._observations, self._states, self._final_observations, self._final_states = self._convert(
            observations, self._wrapper.state(), *self._wrapper.final()
        )
        return self._observations, _torch2warp(reward), _torch2warp(terminated), _torch2warp(truncated), info

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
        self._observations, self._states, self._final_observations, self._final_states = self._convert(
            observations, self._wrapper.state(), *self._wrapper.final()
        )
        return self._observations, info

    def final(self) -> tuple[Tensor, Tensor | None]:
        """Get the observations and states with which the last transitions ended.

        The final observation is read from the info under the first available key (``"final_obs"``,
        ``"final_observation"`` or ``"final_observations"``) when the environment is auto-reset (e.g.:
        ``ManiSkillVectorEnv(auto_reset=True)``). Otherwise, it is the observation reached before the wrapper resets
        the environments whose episodes ended. The wrapper does not provide states, so the final state is the same as
        the one returned by :py:meth:`state`.

        See :py:meth:`~skrl.envs.wrappers.warp.base.Wrapper.final` for more details.

        :return: Final observation, state.
        """
        return self._final_observations, self._final_states

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        return self._wrapper.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment."""
        self._wrapper.close()
