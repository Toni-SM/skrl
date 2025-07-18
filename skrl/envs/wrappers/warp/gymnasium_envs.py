from typing import Any, Tuple, Union

import gymnasium

import numpy as np
import warp as wp

from skrl import logger
from skrl.envs.wrappers.warp.base import Wrapper
from skrl.utils.spaces.warp import (
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

    def step(self, actions: wp.array) -> Tuple[
        wp.array,
        wp.array,
        wp.array,
        wp.array,
        Any,
    ]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: wp.array

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of wp.array and any other info
        """
        actions = untensorize_space(
            self.action_space,
            unflatten_tensorized_space(self.action_space, actions),
            squeeze_batch_dimension=not self._vectorized,
        )

        observation, reward, terminated, truncated, info = self._env.step(actions)

        # convert response to warp
        observation = flatten_tensorized_space(tensorize_space(self.observation_space, observation, device=self.device))
        reward = wp.array(np.array(reward), dtype=wp.float32).reshape((self.num_envs, -1))
        terminated = wp.array(np.array(terminated), dtype=wp.int8).reshape((self.num_envs, -1))
        truncated = wp.array(np.array(truncated), dtype=wp.int8).reshape((self.num_envs, -1))

        # save observation and info for vectorized envs
        if self._vectorized:
            self._observation = observation
            self._info = info

        return observation, reward, terminated, truncated, info

    def state(self) -> Union[wp.array, None]:
        """Get the environment state

        :return: State
        :rtype: wp.array
        """
        try:
            return flatten_tensorized_space(
                tensorize_space(self.state_space, self._unwrapped.state(), device=self.device)
            )
        except:
            return None
        # if hasattr(self._unwrapped, "state"):
        #     return flatten_tensorized_space(
        #         tensorize_space(self.state_space, self._unwrapped.state(), device=self.device)
        #     )
        # return None

    def reset(self) -> Tuple[wp.array, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: wp.array and any other info
        """
        # handle vectorized environments (vector environments are autoreset)
        if self._vectorized:
            if self._reset_once:
                observation, self._info = self._env.reset()
                self._observation = flatten_tensorized_space(
                    tensorize_space(self.observation_space, observation, device=self.device)
                )
                self._reset_once = False
            return self._observation, self._info

        observation, info = self._env.reset()
        observation = flatten_tensorized_space(tensorize_space(self.observation_space, observation, device=self.device))
        return observation, info

    def render(self, *args, **kwargs) -> Any:
        """Render the environment"""
        if self._vectorized:
            return self._env.call("render", *args, **kwargs)
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment"""
        self._env.close()
