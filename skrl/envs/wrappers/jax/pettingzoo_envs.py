from typing import Any, Mapping, Tuple, Union

import collections

import jax
import numpy as np

from skrl.envs.wrappers.jax.base import MultiAgentEnvWrapper
from skrl.utils.spaces.jax import (
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

    def step(self, actions: Mapping[str, Union[np.ndarray, jax.Array]]) -> Tuple[
        Mapping[str, Union[np.ndarray, jax.Array]],
        Mapping[str, Union[np.ndarray, jax.Array]],
        Mapping[str, Union[np.ndarray, jax.Array]],
        Mapping[str, Union[np.ndarray, jax.Array]],
        Mapping[str, Any],
    ]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: dict of np.ndarray or jax.Array

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of dict of np.ndarray or jax.Array and any other info
        """
        if self._jax:
            actions = jax.device_get(actions)
        actions = {
            uid: untensorize_space(self.action_spaces[uid], unflatten_tensorized_space(self.action_spaces[uid], action))
            for uid, action in actions.items()
        }
        observations, rewards, terminated, truncated, infos = self._env.step(actions)

        # convert response to numpy or jax
        observations = {
            uid: flatten_tensorized_space(
                tensorize_space(self.observation_spaces[uid], value, device=self.device, _jax=False), _jax=False
            )
            for uid, value in observations.items()
        }
        rewards = {uid: np.array(value, dtype=np.float32).reshape(self.num_envs, -1) for uid, value in rewards.items()}
        terminated = {
            uid: np.array(value, dtype=np.int8).reshape(self.num_envs, -1) for uid, value in terminated.items()
        }
        truncated = {uid: np.array(value, dtype=np.int8).reshape(self.num_envs, -1) for uid, value in truncated.items()}
        if self._jax:
            observations = {uid: jax.device_put(value, device=self.device) for uid, value in observations.items()}
            rewards = {uid: jax.device_put(value, device=self.device) for uid, value in rewards.items()}
            terminated = {uid: jax.device_put(value, device=self.device) for uid, value in terminated.items()}
            truncated = {uid: jax.device_put(value, device=self.device) for uid, value in truncated.items()}
        return observations, rewards, terminated, truncated, infos

    def state(self) -> Union[np.ndarray, jax.Array]:
        """Get the environment state

        :return: State
        :rtype: np.ndarray or jax.Array
        """
        state = flatten_tensorized_space(
            tensorize_space(next(iter(self.state_spaces.values())), self._env.state(), device=self.device, _jax=False),
            _jax=False,
        )
        if self._jax:
            state = jax.device_put(state, device=self.device)
        return state

    def reset(self) -> Tuple[Mapping[str, Union[np.ndarray, jax.Array]], Mapping[str, Any]]:
        """Reset the environment

        :return: Observation, info
        :rtype: tuple of dict of np.ndarray or jax.Array and any other info
        """
        outputs = self._env.reset()
        if isinstance(outputs, collections.abc.Mapping):
            observations = outputs
            infos = {uid: {} for uid in self.possible_agents}
        else:
            observations, infos = outputs

        # convert response to numpy or jax
        observations = {
            uid: flatten_tensorized_space(
                tensorize_space(self.observation_spaces[uid], value, device=self.device, _jax=False), _jax=False
            )
            for uid, value in observations.items()
        }
        if self._jax:
            observations = {uid: jax.device_put(value, device=self.device) for uid, value in observations.items()}
        return observations, infos

    def render(self, *args, **kwargs) -> Any:
        """Render the environment"""
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment"""
        self._env.close()
