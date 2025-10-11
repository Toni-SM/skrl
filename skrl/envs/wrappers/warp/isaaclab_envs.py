from __future__ import annotations

from typing import Any

import gymnasium

import warp as wp


try:
    import torch
except:
    pass  # TODO: show warning message
from skrl.envs.wrappers.warp.base import Wrapper
from skrl.utils.spaces.warp import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space


class IsaacLabWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Lab environment wrapper.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._reset_once = True
        self._observations = None
        self._states = None
        self._info = {}

    @property
    def state_space(self) -> gymnasium.Space | None:
        """State space."""
        try:
            return self._unwrapped.single_observation_space["critic"]
        except KeyError:
            pass
        try:
            return self._unwrapped.state_space
        except AttributeError:
            return None

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space."""
        try:
            return self._unwrapped.single_observation_space["policy"]
        except:
            return self._unwrapped.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space."""
        try:
            return self._unwrapped.single_action_space
        except:
            return self._unwrapped.action_space

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array, wp.array, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        actions = unflatten_tensorized_space(self.action_space, actions)
        with torch.no_grad():
            observations, reward, terminated, truncated, self._info = self._env.step(wp.to_torch(actions))
        self._observations = flatten_tensorized_space(
            tensorize_space(self.observation_space, wp.from_torch(observations["policy"]))
        )
        states = observations.get("critic", None)
        if states is not None:
            self._states = flatten_tensorized_space(tensorize_space(self.state_space, wp.from_torch(states)))
        return (
            self._observations,
            wp.from_torch(reward.view(-1, 1)),
            wp.from_torch(terminated.view(-1, 1)),
            wp.from_torch(truncated.view(-1, 1)),
            self._info,
        )

    def state(self) -> wp.array | None:
        """Get the environment state.

        :return: State.
        """
        return self._states

    def reset(self) -> tuple[wp.array, Any]:
        """Reset the environment.

        :return: Observation, info.
        """
        if self._reset_once:
            observations, self._info = self._env.reset()
            self._observations = flatten_tensorized_space(
                tensorize_space(self.observation_space, wp.from_torch(observations["policy"]))
            )
            states = observations.get("critic", None)
            if states is not None:
                self._states = flatten_tensorized_space(tensorize_space(self.state_space, wp.from_torch(states)))
            self._reset_once = False
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        return None

    def close(self) -> None:
        """Close the environment."""
        self._env.close()


class IsaacLabMultiAgentWrapper:
    def __init__(self, env: Any) -> None:
        raise NotImplementedError
