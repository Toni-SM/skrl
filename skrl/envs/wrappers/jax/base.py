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
        try:
            self._unwrapped = self._env.unwrapped
        except:
            self._unwrapped = env

        # device
        self._device = None
        if hasattr(self._unwrapped, "device"):
            if type(self._unwrapped.device) == str:
                device_type, device_index = f"{self._unwrapped.device}:0".split(':')[:2]
                try:
                    self._device = jax.devices(device_type)[int(device_index)]
                except (RuntimeError, IndexError):
                    self._device = None
            else:
                self._device = self._unwrapped.device
        if self._device is None:
            self._device = jax.devices()[0]

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
        if hasattr(self._unwrapped, key):
            return getattr(self._unwrapped, key)
        raise AttributeError(f"Wrapped environment ({self._unwrapped.__class__.__name__}) does not have attribute '{key}'")

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

    def state(self) -> Union[np.ndarray, jax.Array]:
        """Get the environment state

        :raises NotImplementedError: Not implemented

        :return: State
        :rtype: np.ndarray or jax.Array
        """
        raise NotImplementedError

    def render(self, *args, **kwargs) -> Any:
        """Render the environment

        :raises NotImplementedError: Not implemented

        :return: Any value from the wrapped environment
        :rtype: any
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the environment

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    @property
    def device(self) -> jax.Device:
        """The device used by the environment

        If the wrapped environment does not have the ``device`` property, the value of this property
        will be ``"cuda"`` or ``"cpu"`` depending on the device availability
        """
        return self._device

    @property
    def num_envs(self) -> int:
        """Number of environments

        If the wrapped environment does not have the ``num_envs`` property, it will be set to 1
        """
        return self._unwrapped.num_envs if hasattr(self._unwrapped, "num_envs") else 1

    @property
    def num_agents(self) -> int:
        """Number of agents

        If the wrapped environment does not have the ``num_agents`` property, it will be set to 1
        """
        return self._unwrapped.num_agents if hasattr(self._unwrapped, "num_agents") else 1

    @property
    def state_space(self) -> Union[gym.Space, None]:
        """State space

        If the wrapped environment does not have the ``state_space`` property, ``None`` will be returned
        """
        return self._unwrapped.state_space if hasattr(self._unwrapped, "state_space") else None

    @property
    def observation_space(self) -> gym.Space:
        """Observation space
        """
        return self._unwrapped.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Action space
        """
        return self._unwrapped.action_space


class MultiAgentEnvWrapper(object):
    def __init__(self, env: Any) -> None:
        """Base wrapper class for multi-agent environments

        :param env: The multi-agent environment to wrap
        :type env: Any supported multi-agent environment
        """
        self._jax = config.jax.backend == "jax"

        self._env = env
        try:
            self._unwrapped = self._env.unwrapped
        except:
            self._unwrapped = env

        # device
        self._device = None
        if hasattr(self._unwrapped, "device"):
            if type(self._unwrapped.device) == str:
                device_type, device_index = f"{self._unwrapped.device}:0".split(':')[:2]
                try:
                    self._device = jax.devices(device_type)[int(device_index)]
                except (RuntimeError, IndexError):
                    self._device = None
            else:
                self._device = self._unwrapped.device
        if self._device is None:
            self._device = jax.devices()[0]

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
        if hasattr(self._unwrapped, key):
            return getattr(self._unwrapped, key)
        raise AttributeError(f"Wrapped environment ({self._unwrapped.__class__.__name__}) does not have attribute '{key}'")

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

    def state(self) -> Union[np.ndarray, jax.Array]:
        """Get the environment state

        :raises NotImplementedError: Not implemented

        :return: State
        :rtype: np.ndarray or jax.Array
        """
        raise NotImplementedError

    def render(self, *args, **kwargs) -> Any:
        """Render the environment

        :raises NotImplementedError: Not implemented

        :return: Any value from the wrapped environment
        :rtype: any
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the environment

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    @property
    def device(self) -> jax.Device:
        """The device used by the environment

        If the wrapped environment does not have the ``device`` property, the value of this property
        will be ``"cuda"`` or ``"cpu"`` depending on the device availability
        """
        return self._device

    @property
    def num_envs(self) -> int:
        """Number of environments

        If the wrapped environment does not have the ``num_envs`` property, it will be set to 1
        """
        return self._unwrapped.num_envs if hasattr(self._unwrapped, "num_envs") else 1

    @property
    def num_agents(self) -> int:
        """Number of current agents

        Read from the length of the ``agents`` property if the wrapped environment doesn't define it
        """
        try:
            return self._unwrapped.num_agents
        except:
            return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        """Number of possible agents the environment could generate

        Read from the length of the ``possible_agents`` property if the wrapped environment doesn't define it
        """
        try:
            return self._unwrapped.max_num_agents
        except:
            return len(self.possible_agents)

    @property
    def agents(self) -> Sequence[str]:
        """Names of all current agents

        These may be changed as an environment progresses (i.e. agents can be added or removed)
        """
        return self._unwrapped.agents

    @property
    def possible_agents(self) -> Sequence[str]:
        """Names of all possible agents the environment could generate

        These can not be changed as an environment progresses
        """
        return self._unwrapped.possible_agents

    @property
    def state_spaces(self) -> Mapping[str, gym.Space]:
        """State spaces

        Since the state space is a global view of the environment (and therefore the same for all the agents),
        this property returns a dictionary (for consistency with the other space-related properties) with the same
        space for all the agents
        """
        space = self._unwrapped.state_space
        return {agent: space for agent in self.possible_agents}

    @property
    def observation_spaces(self) -> Mapping[str, gym.Space]:
        """Observation spaces
        """
        return self._unwrapped.observation_spaces

    @property
    def action_spaces(self) -> Mapping[str, gym.Space]:
        """Action spaces
        """
        return self._unwrapped.action_spaces

    def state_space(self, agent: str) -> gym.Space:
        """State space

        Since the state space is a global view of the environment (and therefore the same for all the agents),
        this method (implemented for consistency with the other space-related methods) returns the same
        space for each queried agent

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
