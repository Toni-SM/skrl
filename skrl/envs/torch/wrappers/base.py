from typing import Tuple, Any

import gym

import torch


class Wrapper(object):
    def __init__(self, env: Any) -> None:
        """Base wrapper class for RL environments

        :param env: The environment to wrap
        :type env: Any supported RL environment
        """
        self._env = env

        # device (faster than @property)
        if hasattr(self._env, "device"):
            self.device = torch.device(self._env.device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getattr__(self, key: str) -> Any:
        """Get an attribute from the wrapped environment

        :param key: The attribute name
        :type key: str

        :raises AttributeError: If the attribute does not exist

        :return: The attribute value
        :rtype: Any
        """
        if hasattr(self._env, key):
            return getattr(self._env, key)
        raise AttributeError("Wrapped environment ({}) does not have attribute '{}'" \
            .format(self._env.__class__.__name__, key))

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :raises NotImplementedError: Not implemented

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        raise NotImplementedError

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :raises NotImplementedError: Not implemented

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        raise NotImplementedError

    def render(self, *args, **kwargs) -> None:
        """Render the environment

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the environment

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    @property
    def num_envs(self) -> int:
        """Number of environments

        If the wrapped environment does not have the ``num_envs`` property, it will be set to 1
        """
        return self._env.num_envs if hasattr(self._env, "num_envs") else 1

    @property
    def state_space(self) -> gym.Space:
        """State space

        If the wrapped environment does not have the ``state_space`` property,
        the value of the ``observation_space`` property will be used
        """
        return self._env.state_space if hasattr(self._env, "state_space") else self._env.observation_space

    @property
    def observation_space(self) -> gym.Space:
        """Observation space
        """
        return self._env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Action space
        """
        return self._env.action_space
