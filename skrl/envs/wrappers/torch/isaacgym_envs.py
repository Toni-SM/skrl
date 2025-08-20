from typing import Any, Tuple, Union

import gymnasium

import torch

from skrl.envs.wrappers.torch.base import Wrapper
from skrl.utils.spaces.torch import (
    convert_gym_space,
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
)


class IsaacGymPreview2Wrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Gym environment (preview 2) wrapper.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._reset_once = True
        self._observations = None
        self._info = {}

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space."""
        return convert_gym_space(self._unwrapped.observation_space)

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space."""
        return convert_gym_space(self._unwrapped.action_space)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        observations, reward, terminated, self._info = self._env.step(
            unflatten_tensorized_space(self.action_space, actions)
        )
        self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations))
        truncated = self._info["time_outs"] if "time_outs" in self._info else torch.zeros_like(terminated)
        return self._observations, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def state(self) -> None:
        """Get the environment state."""
        pass

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment.

        :return: Observation, info.
        """
        if self._reset_once:
            observations = self._env.reset()
            self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations))
            self._reset_once = False
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        return None

    def close(self) -> None:
        """Close the environment."""
        pass


class IsaacGymPreview3Wrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Gym environment (preview 3) wrapper.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._reset_once = True
        self._observations = None
        self._states = None
        self._info = {}

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space."""
        return convert_gym_space(self._unwrapped.observation_space)

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space."""
        return convert_gym_space(self._unwrapped.action_space)

    @property
    def state_space(self) -> Union[gymnasium.Space, None]:
        """State space."""
        try:
            if self.num_states:
                return convert_gym_space(self._unwrapped.state_space)
        except:
            pass
        return None

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        observations, reward, terminated, self._info = self._env.step(
            unflatten_tensorized_space(self.action_space, actions)
        )
        self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations["obs"]))
        truncated = self._info["time_outs"] if "time_outs" in self._info else torch.zeros_like(terminated)
        states = observations.get("states", None)
        if states is not None:
            self._states = flatten_tensorized_space(tensorize_space(self.state_space, states))
        return self._observations, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def state(self) -> Union[torch.Tensor, None]:
        """Get the environment state.

        :return: State.
        """
        return self._states

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment.

        :return: Observation, info.
        """
        if self._reset_once:
            observations = self._env.reset()
            self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations["obs"]))
            states = observations.get("states", None)
            if states is not None:
                self._states = flatten_tensorized_space(tensorize_space(self.state_space, states))
            self._reset_once = False
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        return None

    def close(self) -> None:
        """Close the environment."""
        pass
