from __future__ import annotations

from typing import Any

import gymnasium

import torch

from skrl import config
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space


class ManiSkillWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """ManiSkill environment wrapper.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._seed = config.torch.key
        self._reset_once = True
        self._observations = None
        self._states = None
        self._info = {}

    @property
    def state_space(self) -> gymnasium.Space | None:
        """State space."""
        try:
            return self._unwrapped.single_state_space
        except AttributeError:
            pass
        try:
            return self._unwrapped.state_space
        except AttributeError:
            return None

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space."""
        try:
            return self._unwrapped.single_observation_space
        except:
            return self._unwrapped.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space."""
        try:
            return self._unwrapped.single_action_space
        except:
            return self._unwrapped.action_space

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        actions = unflatten_tensorized_space(self.action_space, actions)

        with torch.no_grad():
            observations, reward, terminated, truncated, self._info = self._env.step(actions)
            # auto-reset environments
            dones = (terminated | truncated).flatten()
            if dones.any():
                env_idx = torch.arange(self.num_envs, device=dones.device)[dones]
                observations, self._info = self._env.reset(options={"env_idx": env_idx})

        self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations))
        return self._observations, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def state(self) -> torch.Tensor | None:
        """Get the environment state.

        :return: State.
        """
        return self._states

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        if self._reset_once:
            observations, self._info = self._env.reset(seed=self._seed)
            self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations))
            self._reset_once = False
            self._seed = None
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
