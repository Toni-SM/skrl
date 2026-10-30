from __future__ import annotations

from typing import Any

import gymnasium

import torch
import warp as wp

from skrl import config

# Isaac Lab runs on PyTorch: reuse its PyTorch wrappers and only convert inputs/outputs (avoid duplicate logic)
from skrl.envs.wrappers.torch.isaaclab_envs import IsaacLabMultiAgentWrapper as _TorchIsaacLabMultiAgentWrapper
from skrl.envs.wrappers.torch.isaaclab_envs import IsaacLabWrapper as _TorchIsaacLabWrapper
from skrl.envs.wrappers.warp.base import MultiAgentEnvWrapper, Wrapper


def _torch2warp(tensor: torch.Tensor | None) -> wp.array | None:
    return None if tensor is None else wp.from_torch(tensor)


def _warp2torch(array: wp.array | None) -> torch.Tensor | None:
    return None if array is None else wp.to_torch(array)


class IsaacLabWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Lab environment wrapper.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._wrapper = _TorchIsaacLabWrapper(env)
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
    ) -> tuple[wp.array, wp.array | None, wp.array, wp.array | None]:
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

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array, wp.array, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        observations, reward, terminated, truncated, info = self._wrapper.step(_warp2torch(actions))
        self._observations, self._states, self._final_observations, self._final_states = self._convert(
            observations, self._wrapper.state(), *self._wrapper.final()
        )
        return self._observations, _torch2warp(reward), _torch2warp(terminated), _torch2warp(truncated), info

    def state(self) -> wp.array | None:
        """Get the environment state.

        :return: State.
        """
        return self._states

    def reset(self) -> tuple[wp.array, Any]:
        """Reset the environment.

        :return: Observation, info.
        """
        observations, info = self._wrapper.reset()
        self._observations, self._states, self._final_observations, self._final_states = self._convert(
            observations, self._wrapper.state(), *self._wrapper.final()
        )
        return self._observations, info

    def final(self) -> tuple[wp.array, wp.array | None]:
        """Get the observations and states with which the last transitions ended.

        The final observations/states are read from the info (``"final_obs"`` key) when the environment
        computes them (i.e.: ``compute_final_obs=True`` in the environment configuration).

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


class IsaacLabMultiAgentWrapper(MultiAgentEnvWrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Lab environment wrapper for multi-agent implementation.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._wrapper = _TorchIsaacLabMultiAgentWrapper(env)
        self._wrapper._seed = config.warp.key
        self._observations = None
        self._states = None
        self._final_observations = None
        self._final_states = None

    def _convert(
        self,
        observations: dict[str, torch.Tensor],
        states: dict[str, torch.Tensor | None],
        final_observations: dict[str, torch.Tensor],
        final_states: dict[str, torch.Tensor | None],
    ) -> tuple[dict[str, wp.array], dict[str, wp.array | None], dict[str, wp.array], dict[str, wp.array | None]]:
        """Convert the PyTorch wrapper's observations/states and final observations/states.

        :param observations: Observations returned by the PyTorch wrapper's :py:meth:`step`/:py:meth:`reset`.
        :param states: States returned by the PyTorch wrapper's :py:meth:`state`.
        :param final_observations: Final observations returned by the PyTorch wrapper's :py:meth:`final`.
        :param final_states: Final states returned by the PyTorch wrapper's :py:meth:`final`.

        :return: Observation, state, final observation, final state.
        """
        converted_observations = {uid: _torch2warp(value) for uid, value in observations.items()}
        # the same state is shared by all agents: convert it once
        converted_state = _torch2warp(next(iter(states.values()), None))
        converted_states = {uid: converted_state for uid in states}
        # avoid converting the same tensor twice when no episode ended
        return (
            converted_observations,
            converted_states,
            {
                uid: converted_observations[uid] if value is observations[uid] else _torch2warp(value)
                for uid, value in final_observations.items()
            },
            (
                converted_states
                if final_states is states
                else {uid: _torch2warp(value) for uid, value in final_states.items()}
            ),
        )

    def step(
        self, actions: dict[str, wp.array]
    ) -> tuple[dict[str, wp.array], dict[str, wp.array], dict[str, wp.array], dict[str, wp.array], dict[str, Any]]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        observations, rewards, terminated, truncated, info = self._wrapper.step(
            {uid: _warp2torch(value) for uid, value in actions.items()}
        )
        self._observations, self._states, self._final_observations, self._final_states = self._convert(
            observations, self._wrapper.state(), *self._wrapper.final()
        )
        return (
            self._observations,
            {uid: _torch2warp(value) for uid, value in rewards.items()},
            {uid: _torch2warp(value) for uid, value in terminated.items()},
            {uid: _torch2warp(value) for uid, value in truncated.items()},
            info,
        )

    def reset(self) -> tuple[dict[str, wp.array], dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        observations, info = self._wrapper.reset()
        self._observations, self._states, self._final_observations, self._final_states = self._convert(
            observations, self._wrapper.state(), *self._wrapper.final()
        )
        return self._observations, info

    def state(self) -> dict[str, wp.array | None]:
        """Get the environment state.

        :return: State.
        """
        return self._states

    def final(self) -> tuple[dict[str, wp.array], dict[str, wp.array | None]]:
        """Get the observations and states with which the last transitions ended.

        The final observations are read from each agent's info (``"final_obs"`` key) when the environment
        computes them (i.e.: ``compute_final_obs=True`` in the environment configuration). The final state is
        not provided by the environment, so it is the same as the one returned by :py:meth:`state`.

        See :py:meth:`~skrl.envs.wrappers.warp.base.MultiAgentEnvWrapper.final` for more details.

        :return: Final observation, state.
        """
        return self._final_observations, self._final_states

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        return self._wrapper.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment."""
        self._wrapper.close()
