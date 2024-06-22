from typing import Any, Mapping, Sequence, Tuple, Union

import gym

import jax
import numpy as np

from skrl import config


class Wrapper(object):
    def __init__(self, env: Any) -> None:
        """Base wrapper class for RL environments

        :param env: The environment to wrap
        :type env: Any supported RL environment
        """
        self._jax = config.jax.backend == "jax"

        self._env = env

        # device (faster than @property)
        self.device = None
        if hasattr(self._env, "device"):
            if type(self._env.device) == str:
                device_type, device_index = f"{self._env.device}:0".split(':')[:2]
                try:
                    self.device = jax.devices(device_type)[int(device_index)]
                except RuntimeError:
                    self.device = None
            else:
                self.device = self._env.device
        if self.device is None:
            self.device = jax.devices()[0]

        # spaces
        try:
            self._action_space = self._env.single_action_space
            self._observation_space = self._env.single_observation_space
        except AttributeError:
            self._action_space = self._env.action_space
            self._observation_space = self._env.observation_space
        self._state_space = self._env.state_space if hasattr(self._env, "state_space") else self._observation_space

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
        raise AttributeError(f"Wrapped environment ({self._env.__class__.__name__}) does not have attribute '{key}'")

    def reset(self) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """Reset the environment

        :raises NotImplementedError: Not implemented

        :return: Observation, info
        :rtype: np.ndarray or jax.Array and any other info
        """
        raise NotImplementedError

    def step(self, actions: Union[np.ndarray, jax.Array]) -> \
        Tuple[Union[np.ndarray, jax.Array], Union[np.ndarray, jax.Array],
              Union[np.ndarray, jax.Array], Union[np.ndarray, jax.Array], Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: np.ndarray or jax.Array

        :raises NotImplementedError: Not implemented

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of np.ndarray or jax.Array and any other info
        """
        raise NotImplementedError

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        pass

    def close(self) -> None:
        """Close the environment
        """
        pass

    @property
    def num_envs(self) -> int:
        """Number of environments

        If the wrapped environment does not have the ``num_envs`` property, it will be set to 1
        """
        return self._env.num_envs if hasattr(self._env, "num_envs") else 1

    @property
    def num_agents(self) -> int:
        """Number of agents

        If the wrapped environment does not have the ``num_agents`` property, it will be set to 1
        """
        return self._env.num_agents if hasattr(self._env, "num_agents") else 1

    @property
    def state_space(self) -> gym.Space:
        """State space

        If the wrapped environment does not have the ``state_space`` property,
        the value of the ``observation_space`` property will be used
        """
        return self._state_space

    @property
    def observation_space(self) -> gym.Space:
        """Observation space
        """
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        """Action space
        """
        return self._action_space


class MultiAgentEnvWrapper(object):
    def __init__(self, env: Any) -> None:
        """Base wrapper class for multi-agent environments

        :param env: The multi-agent environment to wrap
        :type env: Any supported multi-agent environment
        """
        self._jax = config.jax.backend == "jax"

        self._env = env

        # device (faster than @property)
        self.device = None
        if hasattr(self._env, "device"):
            if type(self._env.device) == str:
                device_type, device_index = f"{self._env.device}:0".split(':')[:2]
                try:
                    self.device = jax.devices(device_type)[int(device_index)]
                except RuntimeError:
                    self.device = None
            else:
                self.device = self._env.device
        if self.device is None:
            self.device = jax.devices()[0]

        self.possible_agents = []

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
        raise AttributeError(f"Wrapped environment ({self._env.__class__.__name__}) does not have attribute '{key}'")

    def reset(self) -> Tuple[Mapping[str, Union[np.ndarray, jax.Array]], Mapping[str, Any]]:
        """Reset the environment

        :raises NotImplementedError: Not implemented

        :return: Observation, info
        :rtype: tuple of dict of np.ndarray or jax.Array and any other info
        """
        raise NotImplementedError

    def step(self, actions: Mapping[str, Union[np.ndarray, jax.Array]]) -> \
        Tuple[Mapping[str, Union[np.ndarray, jax.Array]], Mapping[str, Union[np.ndarray, jax.Array]],
              Mapping[str, Union[np.ndarray, jax.Array]], Mapping[str, Union[np.ndarray, jax.Array]],
              Mapping[str, Any]]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: dict of np.ndarray or jax.Array

        :raises NotImplementedError: Not implemented

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of dict of np.ndarray or jax.Array and any other info
        """
        raise NotImplementedError

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        pass

    def close(self) -> None:
        """Close the environment
        """
        pass

    @property
    def num_envs(self) -> int:
        """Number of environments

        If the wrapped environment does not have the ``num_envs`` property, it will be set to 1
        """
        return self._env.num_envs if hasattr(self._env, "num_envs") else 1

    @property
    def num_agents(self) -> int:
        """Number of agents

        If the wrapped environment does not have the ``num_agents`` property, it will be set to 1
        """
        return self._env.num_agents if hasattr(self._env, "num_agents") else 1

    @property
    def agents(self) -> Sequence[str]:
        """Names of all current agents

        These may be changed as an environment progresses (i.e. agents can be added or removed)
        """
        raise NotImplementedError

    @property
    def state_spaces(self) -> Mapping[str, gym.Space]:
        """State spaces

        An alias for the ``observation_spaces`` property
        """
        return self.observation_spaces

    @property
    def observation_spaces(self) -> Mapping[str, gym.Space]:
        """Observation spaces
        """
        raise NotImplementedError

    @property
    def action_spaces(self) -> Mapping[str, gym.Space]:
        """Action spaces
        """
        raise NotImplementedError

    @property
    def shared_state_spaces(self) -> Mapping[str, gym.Space]:
        """Shared state spaces

        An alias for the ``shared_observation_spaces`` property
        """
        return self.shared_observation_spaces

    @property
    def shared_observation_spaces(self) -> Mapping[str, gym.Space]:
        """Shared observation spaces
        """
        raise NotImplementedError

    def state_space(self, agent: str) -> gym.Space:
        """State space

        :param agent: Name of the agent
        :type agent: str

        :return: The state space for the specified agent
        :rtype: gym.Space
        """
        return self.state_spaces[agent]

    def observation_space(self, agent: str) -> gym.Space:
        """Observation space

        :param agent: Name of the agent
        :type agent: str

        :return: The observation space for the specified agent
        :rtype: gym.Space
        """
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        """Action space

        :param agent: Name of the agent
        :type agent: str

        :return: The action space for the specified agent
        :rtype: gym.Space
        """
        return self.action_spaces[agent]

    def shared_state_space(self, agent: str) -> gym.Space:
        """Shared state space

        :param agent: Name of the agent
        :type agent: str

        :return: The shared state space for the specified agent
        :rtype: gym.Space
        """
        return self.shared_state_spaces[agent]

    def shared_observation_space(self, agent: str) -> gym.Space:
        """Shared observation space

        :param agent: Name of the agent
        :type agent: str

        :return: The shared observation space for the specified agent
        :rtype: gym.Space
        """
        return self.shared_observation_spaces[agent]
