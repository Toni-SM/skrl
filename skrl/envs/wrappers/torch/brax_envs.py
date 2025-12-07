from __future__ import annotations

from typing import Any

import gymnasium

import torch

from skrl import logger
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.utils.spaces.torch import (
    convert_gym_space,
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
)


class BraxWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Brax environment wrapper.

        :param env: The environment instance to wrap.
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
        """Observation space."""
        return convert_gym_space(self._unwrapped.observation_space, squeeze_batch_dimension=True)

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space."""
        return convert_gym_space(self._unwrapped.action_space, squeeze_batch_dimension=True)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        observation, reward, terminated, info = self._env.step(unflatten_tensorized_space(self.action_space, actions))
        observation = flatten_tensorized_space(tensorize_space(self.observation_space, observation))
        truncated = torch.zeros_like(terminated)
        return observation, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), info

    def state(self) -> torch.Tensor | None:
        """Get the environment state.

        :return: State.
        """
        try:
            return flatten_tensorized_space(
                tensorize_space(self.state_space, self._unwrapped.state(), device=self.device)
            )
        except:
            return None

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        observation = self._env.reset()
        observation = flatten_tensorized_space(tensorize_space(self.observation_space, observation))
        return observation, {}

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        frame = self._env.render(mode="rgb_array")
        frame = frame[0] if frame.ndim == 4 else frame

        # render the frame using OpenCV
        try:
            import cv2

            cv2.imshow("env", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
        except ImportError as e:
            logger.warning(f"Unable to import opencv-python: {e}. Frame will not be rendered.")
        return frame

    def close(self) -> None:
        """Close the environment."""
        try:
            self._env.close()
        except AttributeError:  # 'VectorGymWrapper' object has no attribute 'closed'
            pass
