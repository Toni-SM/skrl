from typing import Any, Tuple, Union

import gymnasium

import jax
import numpy as np

from skrl import logger
from skrl.envs.wrappers.jax.base import Wrapper
from skrl.utils.spaces.jax import (
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
    untensorize_space,
)


class GymnasiumWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Gymnasium environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Gymnasium environment
        """
        super().__init__(env)

        self._vectorized = False
        try:
            self._vectorized = self._vectorized or isinstance(env, gymnasium.vector.VectorEnv)
        except Exception as e:
            pass
        try:
            self._vectorized = self._vectorized or isinstance(env, gymnasium.experimental.vector.VectorEnv)
        except Exception as e:
            logger.warning(f"Failed to check for a vectorized environment: {e}")
        if self._vectorized:
            self._reset_once = True
            self._observation = None
            self._info = None

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space"""
        if self._vectorized:
            return self._env.single_observation_space
        return self._env.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space"""
        if self._vectorized:
            return self._env.single_action_space
        return self._env.action_space

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

        observation, reward, terminated, truncated, info = self._env.step(actions)

        # convert response to numpy or jax
        observation = flatten_tensorized_space(
            tensorize_space(self.observation_space, observation, device=self.device, _jax=False), _jax=False
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
                observation, self._info = self._env.reset()
                self._observation = flatten_tensorized_space(
                    tensorize_space(self.observation_space, observation, device=self.device, _jax=False), _jax=False
                )
                if self._jax:
                    self._observation = jax.device_put(self._observation, device=self.device)
                self._reset_once = False
            return self._observation, self._info

        observation, info = self._env.reset()

        # convert response to numpy or jax
        observation = flatten_tensorized_space(
            tensorize_space(self.observation_space, observation, device=self.device, _jax=False), _jax=False
        )
        if self._jax:
            observation = jax.device_put(observation, device=self.device)
        return observation, info

    def render(self, *args, **kwargs) -> Any:
        """Render the environment"""
        if self._vectorized:
            return self._env.call("render", *args, **kwargs)
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment"""
        self._env.close()
