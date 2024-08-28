from typing import Any, Tuple

import gymnasium

import numpy as np
import torch

from skrl import logger
from skrl.envs.wrappers.torch.base import Wrapper


class BraxWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Brax environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Brax environment
        """
        super().__init__(env)

        import brax.envs.wrappers.gym
        import brax.envs.wrappers.torch
        env = brax.envs.wrappers.gym.VectorGymWrapper(env)
        env = brax.envs.wrappers.torch.TorchWrapper(env, device=self.device)
        self._env = env
        self._unwrapped = env.unwrapped

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space
        """
        limit = np.inf * np.ones(self._unwrapped.observation_space.shape[1:], dtype='float32')
        return gymnasium.spaces.Box(-limit, limit, dtype='float32')

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space
        """
        limit = np.inf * np.ones(self._unwrapped.action_space.shape[1:], dtype='float32')
        return gymnasium.spaces.Box(-limit, limit, dtype='float32')

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        observation, reward, terminated, info = self._env.step(actions)
        truncated = torch.zeros_like(terminated)
        return observation, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        observation = self._env.reset()
        return observation, {}

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        frame = self._env.render(mode="rgb_array")

        # render the frame using OpenCV
        try:
            import cv2
            cv2.imshow("env", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
        except ImportError as e:
            logger.warning(f"Unable to import opencv-python: {e}. Frame will not be rendered.")
        return frame

    def close(self) -> None:
        """Close the environment
        """
        # self._env.close() raises AttributeError: 'VectorGymWrapper' object has no attribute 'closed'
        pass
