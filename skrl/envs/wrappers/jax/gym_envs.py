from typing import Any, Tuple, Union

import gymnasium
from packaging import version

import jax
import numpy as np

from skrl import logger
from skrl.envs.wrappers.jax.base import Wrapper
from skrl.utils.spaces.jax import (
    convert_gym_space,
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
    untensorize_space,
)


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

        import gym

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
        """Observation space"""
        if self._vectorized:
            return convert_gym_space(self._env.single_observation_space)
        return convert_gym_space(self._env.observation_space)

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space"""
        if self._vectorized:
            return convert_gym_space(self._env.single_action_space)
        return convert_gym_space(self._env.action_space)

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
        if self._jax or isinstance(actions, jax.Array):
            actions = np.asarray(jax.device_get(actions))
        actions = untensorize_space(
            self.action_space,
            unflatten_tensorized_space(self.action_space, actions),
            squeeze_batch_dimension=not self._vectorized,
        )

        if self._deprecated_api:
            observation, reward, terminated, info = self._env.step(actions)
            # truncated: https://gymnasium.farama.org/tutorials/handling_time_limits
            if type(info) is list:
                truncated = np.array([d.get("TimeLimit.truncated", False) for d in info], dtype=terminated.dtype)
                terminated *= np.logical_not(truncated)
            else:
                truncated = info.get("TimeLimit.truncated", False)
                if truncated:
                    terminated = False
        else:
            observation, reward, terminated, truncated, info = self._env.step(actions)

        # convert response to numpy or jax
        observation = flatten_tensorized_space(
            tensorize_space(self.observation_space, observation, self.device, False), False
        )
        reward = np.array(reward, dtype=np.float32).reshape(self.num_envs, -1)
        terminated = np.array(terminated, dtype=np.int8).reshape(self.num_envs, -1)
        truncated = np.array(truncated, dtype=np.int8).reshape(self.num_envs, -1)
        if self._jax:
            observation = jax.device_put(observation, device=self.device)
            reward = jax.device_put(reward, device=self.device)
            terminated = jax.device_put(terminated, device=self.device)
            truncated = jax.device_put(truncated, device=self.device)

        # save observation and info for vectorized envs
        if self._vectorized:
            self._observation = observation
            self._info = info

        return observation, reward, terminated, truncated, info

    def reset(self) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: np.ndarray or jax.Array and any other info
        """
        # handle vectorized environments (vector environments are autoreset)
        if self._vectorized:
            if self._reset_once:
                if self._deprecated_api:
                    observation = self._env.reset()
                    self._info = {}
                else:
                    observation, self._info = self._env.reset()
                self._observation = flatten_tensorized_space(
                    tensorize_space(self.observation_space, observation, self.device, False), False
                )
                if self._jax:
                    self._observation = jax.device_put(self._observation, device=self.device)
                self._reset_once = False
            return self._observation, self._info

        if self._deprecated_api:
            observation = self._env.reset()
            info = {}
        else:
            observation, info = self._env.reset()

        # convert response to numpy or jax
        observation = flatten_tensorized_space(
            tensorize_space(self.observation_space, observation, self.device, False), False
        )
        if self._jax:
            observation = jax.device_put(observation, device=self.device)
        return observation, info

    def render(self, *args, **kwargs) -> Any:
        """Render the environment"""
        if self._vectorized:
            return None
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment"""
        self._env.close()
