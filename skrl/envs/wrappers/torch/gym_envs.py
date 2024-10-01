from typing import Any, Optional, Tuple

import gym
import gymnasium
from packaging import version

import numpy as np
import torch

from skrl import logger
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.utils.spaces.torch import convert_gym_space, flatten_tensorized_space, tensorize_space


class GymWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """OpenAI Gym environment wrapper

        :param env: The environment to wrap
        :type env: Any supported OpenAI Gym environment
        """
        super().__init__(env)

        # hack to fix: module 'numpy' has no attribute 'bool8'
        try:
            np.bool8
        except AttributeError:
            np.bool8 = np.bool

        self._vectorized = False
        try:
            if isinstance(env, gym.vector.VectorEnv):
                self._vectorized = True
                self._reset_once = True
                self._observation = None
                self._info = None
        except Exception as e:
            logger.warning(f"Failed to check for a vectorized environment: {e}")

        self._deprecated_api = version.parse(gym.__version__) < version.parse("0.25.0")
        if self._deprecated_api:
            logger.warning(f"Using a deprecated version of OpenAI Gym's API: {gym.__version__}")

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space
        """
        if self._vectorized:
            return convert_gym_space(self._env.single_observation_space)
        return convert_gym_space(self._env.observation_space)

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space
        """
        if self._vectorized:
            return convert_gym_space(self._env.single_action_space)
        return convert_gym_space(self._env.action_space)

    def _tensor_to_action(self, actions: torch.Tensor) -> Any:
        """Convert the action to the OpenAI Gym expected format

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :raise ValueError: If the action space type is not supported

        :return: The action in the OpenAI Gym format
        :rtype: Any supported OpenAI Gym action space
        """
        space = convert_gym_space(self._env.action_space) if self._vectorized else self.action_space

        if self._vectorized:
            if isinstance(space, gymnasium.spaces.MultiDiscrete):
                return np.array(actions.cpu().numpy(), dtype=space.dtype).reshape(space.shape)
            elif isinstance(space, gymnasium.spaces.Tuple):
                if isinstance(space[0], gymnasium.spaces.Box):
                    return np.array(actions.cpu().numpy(), dtype=space[0].dtype).reshape(space.shape)
                elif isinstance(space[0], gymnasium.spaces.Discrete):
                    return np.array(actions.cpu().numpy(), dtype=space[0].dtype).reshape(-1)
        if isinstance(space, gymnasium.spaces.Discrete):
            return actions.item()
        elif isinstance(space, gymnasium.spaces.MultiDiscrete):
            return np.array(actions.cpu().numpy(), dtype=space.dtype).reshape(space.shape)
        elif isinstance(space, gymnasium.spaces.Box):
            return np.array(actions.cpu().numpy(), dtype=space.dtype).reshape(space.shape)
        raise ValueError(f"Action space type {type(space)} not supported. Please report this issue")

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        if self._deprecated_api:
            observation, reward, terminated, info = self._env.step(self._tensor_to_action(actions))
            # truncated: https://gymnasium.farama.org/tutorials/handling_time_limits
            if type(info) is list:
                truncated = np.array([d.get("TimeLimit.truncated", False) for d in info], dtype=terminated.dtype)
                terminated *= np.logical_not(truncated)
            else:
                truncated = info.get("TimeLimit.truncated", False)
                if truncated:
                    terminated = False
        else:
            observation, reward, terminated, truncated, info = self._env.step(self._tensor_to_action(actions))

        # convert response to torch
        observation = flatten_tensorized_space(tensorize_space(self.observation_space, observation, self.device))
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        terminated = torch.tensor(terminated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)
        truncated = torch.tensor(truncated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)

        # save observation and info for vectorized envs
        if self._vectorized:
            self._observation = observation
            self._info = info

        return observation, reward, terminated, truncated, info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        # handle vectorized environments (vector environments are autoreset)
        if self._vectorized:
            if self._reset_once:
                if self._deprecated_api:
                    observation = self._env.reset()
                    self._info = {}
                else:
                    observation, self._info = self._env.reset()
                self._observation = flatten_tensorized_space(tensorize_space(self.observation_space, observation, self.device))
                self._reset_once = False
            return self._observation, self._info

        if self._deprecated_api:
            observation = self._env.reset()
            info = {}
        else:
            observation, info = self._env.reset()
        observation = flatten_tensorized_space(tensorize_space(self.observation_space, observation, self.device))
        return observation, info

    def render(self, *args, **kwargs) -> Any:
        """Render the environment
        """
        if self._vectorized:
            return None
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment
        """
        self._env.close()
