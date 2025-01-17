from typing import Any, Tuple, Union

import gymnasium

import jax
import jax.numpy as jnp
import numpy as np

from skrl import logger
from skrl.envs.wrappers.jax.base import Wrapper
from skrl.utils.spaces.jax import (
    convert_gym_space,
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
)


class BraxWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Brax environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Brax environment
        """
        super().__init__(env)

        import brax.envs.wrappers.gym

        env = brax.envs.wrappers.gym.VectorGymWrapper(env)
        self._env = env
        self._unwrapped = env.unwrapped

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space"""
        return convert_gym_space(self._unwrapped.observation_space, squeeze_batch_dimension=True)

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space"""
        return convert_gym_space(self._unwrapped.action_space, squeeze_batch_dimension=True)

    def step(self, actions: Union[np.ndarray, jax.Array]) -> Tuple[
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Any,
    ]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: np.ndarray or jax.Array

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of np.ndarray or jax.Array and any other info
        """
        observation, reward, terminated, info = self._env.step(unflatten_tensorized_space(self.action_space, actions))
        observation = flatten_tensorized_space(tensorize_space(self.observation_space, observation, self.device))
        truncated = jnp.zeros_like(terminated)
        if not self._jax:
            observation = np.asarray(jax.device_get(observation))
            reward = np.asarray(jax.device_get(reward))
            terminated = np.asarray(jax.device_get(terminated))
            truncated = np.asarray(jax.device_get(truncated))
        return observation, reward.reshape(-1, 1), terminated.reshape(-1, 1), truncated.reshape(-1, 1), info

    def reset(self) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: np.ndarray or jax.Array and any other info
        """
        observation = self._env.reset()
        observation = flatten_tensorized_space(tensorize_space(self.observation_space, observation, self.device))
        if not self._jax:
            observation = np.asarray(jax.device_get(observation))
        return observation, {}

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
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
        """Close the environment"""
        # self._env.close() raises AttributeError: 'VectorGymWrapper' object has no attribute 'closed'
        pass
