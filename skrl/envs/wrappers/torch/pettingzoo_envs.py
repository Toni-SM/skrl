from typing import Any, Mapping, Tuple

import collections

import torch

from skrl.envs.wrappers.torch.base import MultiAgentEnvWrapper
from skrl.utils.spaces.torch import (
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
    untensorize_space,
)


class PettingZooWrapper(MultiAgentEnvWrapper):
    def __init__(self, env: Any) -> None:
        """PettingZoo (parallel) environment wrapper

        :param env: The environment to wrap
        :type env: Any supported PettingZoo (parallel) environment
        """
        super().__init__(env)

    def step(self, actions: Mapping[str, torch.Tensor]) -> Tuple[
        Mapping[str, torch.Tensor],
        Mapping[str, torch.Tensor],
        Mapping[str, torch.Tensor],
        Mapping[str, torch.Tensor],
        Mapping[str, Any],
    ]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: dictionary of torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of dictionaries torch.Tensor and any other info
        """
        actions = {
            uid: untensorize_space(self.action_spaces[uid], unflatten_tensorized_space(self.action_spaces[uid], action))
            for uid, action in actions.items()
        }
        observations, rewards, terminated, truncated, infos = self._env.step(actions)

        # convert response to torch
        observations = {
            uid: flatten_tensorized_space(tensorize_space(self.observation_spaces[uid], value, device=self.device))
            for uid, value in observations.items()
        }
        rewards = {
            uid: torch.tensor(value, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
            for uid, value in rewards.items()
        }
        terminated = {
            uid: torch.tensor(value, device=self.device, dtype=torch.bool).view(self.num_envs, -1)
            for uid, value in terminated.items()
        }
        truncated = {
            uid: torch.tensor(value, device=self.device, dtype=torch.bool).view(self.num_envs, -1)
            for uid, value in truncated.items()
        }
        return observations, rewards, terminated, truncated, infos

    def state(self) -> torch.Tensor:
        """Get the environment state

        :return: State
        :rtype: torch.Tensor
        """
        return flatten_tensorized_space(
            tensorize_space(next(iter(self.state_spaces.values())), self._env.state(), device=self.device)
        )

    def reset(self) -> Tuple[Mapping[str, torch.Tensor], Mapping[str, Any]]:
        """Reset the environment

        :return: Observation, info
        :rtype: tuple of dictionaries of torch.Tensor and any other info
        """
        outputs = self._env.reset()
        if isinstance(outputs, collections.abc.Mapping):
            observations = outputs
            infos = {uid: {} for uid in self.possible_agents}
        else:
            observations, infos = outputs

        # convert response to torch
        observations = {
            uid: flatten_tensorized_space(tensorize_space(self.observation_spaces[uid], value, device=self.device))
            for uid, value in observations.items()
        }
        return observations, infos

    def render(self, *args, **kwargs) -> Any:
        """Render the environment"""
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment"""
        self._env.close()
