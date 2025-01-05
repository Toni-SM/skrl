from typing import Any, Mapping, Tuple, Union

import gymnasium

import torch

from skrl.envs.wrappers.torch.base import MultiAgentEnvWrapper, Wrapper
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space


class IsaacLabWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Lab environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Lab environment
        """
        super().__init__(env)

        self._reset_once = True
        self._observations = None
        self._info = {}

    @property
    def state_space(self) -> Union[gymnasium.Space, None]:
        """State space"""
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
        """Observation space"""
        try:
            return self._unwrapped.single_observation_space["policy"]
        except:
            return self._unwrapped.observation_space["policy"]

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space"""
        try:
            return self._unwrapped.single_action_space
        except:
            return self._unwrapped.action_space

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        actions = unflatten_tensorized_space(self.action_space, actions)
        observations, reward, terminated, truncated, self._info = self._env.step(actions)
        self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations["policy"]))
        return self._observations, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        if self._reset_once:
            observations, self._info = self._env.reset()
            self._observations = flatten_tensorized_space(
                tensorize_space(self.observation_space, observations["policy"])
            )
            self._reset_once = False
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        return None

    def close(self) -> None:
        """Close the environment"""
        self._env.close()


class IsaacLabMultiAgentWrapper(MultiAgentEnvWrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Lab environment wrapper for multi-agent implementation

        :param env: The environment to wrap
        :type env: Any supported Isaac Lab environment
        """
        super().__init__(env)

        self._reset_once = True
        self._observations = None
        self._info = {}

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
        actions = {k: unflatten_tensorized_space(self.action_spaces[k], v) for k, v in actions.items()}
        observations, rewards, terminated, truncated, self._info = self._env.step(actions)
        self._observations = {
            k: flatten_tensorized_space(tensorize_space(self.observation_spaces[k], v)) for k, v in observations.items()
        }
        return (
            self._observations,
            {k: v.view(-1, 1) for k, v in rewards.items()},
            {k: v.view(-1, 1) for k, v in terminated.items()},
            {k: v.view(-1, 1) for k, v in truncated.items()},
            self._info,
        )

    def reset(self) -> Tuple[Mapping[str, torch.Tensor], Mapping[str, Any]]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        if self._reset_once:
            observations, self._info = self._env.reset()
            self._observations = {
                k: flatten_tensorized_space(tensorize_space(self.observation_spaces[k], v))
                for k, v in observations.items()
            }
            self._reset_once = False
        return self._observations, self._info

    def state(self) -> torch.Tensor:
        """Get the environment state

        :return: State
        :rtype: torch.Tensor
        """
        try:
            state = self._env.state()
        except AttributeError:  # 'OrderEnforcing' object has no attribute 'state'
            state = self._unwrapped.state()
        if state is not None:
            return flatten_tensorized_space(tensorize_space(next(iter(self.state_spaces.values())), state))
        return state

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        return None

    def close(self) -> None:
        """Close the environment"""
        self._env.close()
