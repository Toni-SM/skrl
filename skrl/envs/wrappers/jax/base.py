from __future__ import annotations

from typing import Any

from abc import ABC, abstractmethod
import gymnasium

import jax
import numpy as np

from skrl import config


class Wrapper(ABC):
    def __init__(self, env: Any) -> None:
        """Base wrapper class for RL environments.

        :param env: The environment instance to wrap.
        """
        self._jax = config.jax.backend == "jax"

        self._env = env
        try:
            self._unwrapped = self._env.unwrapped
        except:
            self._unwrapped = env

        # device
        if hasattr(self._unwrapped, "device"):
            self._device = config.jax.parse_device(self._unwrapped.device)
        else:
            self._device = config.jax.parse_device(None)

    def __getattr__(self, key: str) -> Any:
        """Get an attribute from the wrapped environment.

        :param key: The attribute name.

        :return: The attribute value.

        :raises AttributeError: If the attribute does not exist.
        """
        if hasattr(self._env, key):
            return getattr(self._env, key)
        if hasattr(self._unwrapped, key):
            return getattr(self._unwrapped, key)
        raise AttributeError(
            f"Wrapped environment ({self._unwrapped.__class__.__name__}) does not have attribute '{key}'"
        )

    @abstractmethod
    def reset(self) -> tuple[np.ndarray | jax.Array, dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        pass

    @abstractmethod
    def step(self, actions: np.ndarray | jax.Array) -> tuple[
        np.ndarray | jax.Array,
        np.ndarray | jax.Array,
        np.ndarray | jax.Array,
        np.ndarray | jax.Array,
        Any,
    ]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        pass

    @abstractmethod
    def state(self) -> np.ndarray | jax.Array | None:
        """Get the environment state.

        :return: State.
        """
        pass

    @abstractmethod
    def render(self, *args, **kwargs) -> Any:
        """Render the environment.

        :return: Any value from the wrapped environment.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the environment."""
        pass

    @property
    def device(self) -> jax.Device:
        """The device used by the environment.

        If the wrapped environment does not have the ``device`` property, the value of this property
        will be ``"cuda"`` or ``"cpu"`` depending on the device availability.
        """
        return self._device

    @property
    def num_envs(self) -> int:
        """Number of environments.

        If the wrapped environment does not have the ``num_envs`` property, it will be set to 1.
        """
        return self._unwrapped.num_envs if hasattr(self._unwrapped, "num_envs") else 1

    @property
    def num_agents(self) -> int:
        """Number of agents.

        If the wrapped environment does not have the ``num_agents`` property, it will be set to 1.
        """
        return self._unwrapped.num_agents if hasattr(self._unwrapped, "num_agents") else 1

    @property
    def state_space(self) -> gymnasium.Space | None:
        """State space.

        If the wrapped environment does not have the ``state_space`` property, ``None`` will be returned.
        """
        return self._unwrapped.state_space if hasattr(self._unwrapped, "state_space") else None

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space."""
        return self._unwrapped.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space."""
        return self._unwrapped.action_space


class MultiAgentEnvWrapper(ABC):
    def __init__(self, env: Any) -> None:
        """Base wrapper class for multi-agent environments.

        :param env: The multi-agent environment instance to wrap.
        """
        self._jax = config.jax.backend == "jax"

        self._env = env
        try:
            self._unwrapped = self._env.unwrapped
        except:
            self._unwrapped = env

        # device
        if hasattr(self._unwrapped, "device"):
            self._device = config.jax.parse_device(self._unwrapped.device)
        else:
            self._device = config.jax.parse_device(None)

    def __getattr__(self, key: str) -> Any:
        """Get an attribute from the wrapped environment.

        :param key: The attribute name.

        :return: The attribute value.

        :raises AttributeError: If the attribute does not exist.
        """
        if hasattr(self._env, key):
            return getattr(self._env, key)
        if hasattr(self._unwrapped, key):
            return getattr(self._unwrapped, key)
        raise AttributeError(
            f"Wrapped environment ({self._unwrapped.__class__.__name__}) does not have attribute '{key}'"
        )

    @abstractmethod
    def reset(self) -> tuple[dict[str, np.ndarray | jax.Array], dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        pass

    @abstractmethod
    def step(self, actions: dict[str, np.ndarray | jax.Array]) -> tuple[
        dict[str, np.ndarray | jax.Array],
        dict[str, np.ndarray | jax.Array],
        dict[str, np.ndarray | jax.Array],
        dict[str, np.ndarray | jax.Array],
        dict[str, Any],
    ]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        pass

    @abstractmethod
    def state(self) -> dict[np.ndarray | jax.Array | None]:
        """Get the environment state.

        :return: State.
        """
        pass

    @abstractmethod
    def render(self, *args, **kwargs) -> Any:
        """Render the environment.

        :return: Any value from the wrapped environment.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the environment."""
        pass

    @property
    def device(self) -> jax.Device:
        """The device used by the environment.

        If the wrapped environment does not have the ``device`` property, the value of this property
        will be ``"cuda"`` or ``"cpu"`` depending on the device availability.
        """
        return self._device

    @property
    def num_envs(self) -> int:
        """Number of environments.

        If the wrapped environment does not have the ``num_envs`` property, it will be set to 1.
        """
        return self._unwrapped.num_envs if hasattr(self._unwrapped, "num_envs") else 1

    @property
    def num_agents(self) -> int:
        """Number of current agents.

        Read from the length of the ``agents`` property if the wrapped environment doesn't define it.
        """
        try:
            return self._unwrapped.num_agents
        except:
            return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        """Number of possible agents the environment could generate.

        Read from the length of the ``possible_agents`` property if the wrapped environment doesn't define it.
        """
        try:
            return self._unwrapped.max_num_agents
        except:
            return len(self.possible_agents)

    @property
    def agents(self) -> list[str]:
        """Names of all current agents.

        These may be changed as an environment progresses (i.e. agents can be added or removed).
        """
        return self._unwrapped.agents

    @property
    def possible_agents(self) -> list[str]:
        """Names of all possible agents the environment could generate.

        These can not be changed as an environment progresses.
        """
        return self._unwrapped.possible_agents

    @property
    def state_spaces(self) -> dict[str, gymnasium.Space | None]:
        """State spaces.

        Although this property returns a dictionary, the space for each agent adheres to the next rules:

        * The wrapped environment has the ``state_space`` attribute (homogeneous state).
          The state is a global view of the environment, so the space is the same for all agents.
        * The wrapped environment has the ``state_spaces`` attribute (heterogeneous state).
          The state may differ for each agent, so the agent spaces may also differ.
        * The wrapped environment does not have the previous attributes. The space is ``None`` for all agents.
        """
        if hasattr(self._unwrapped, "state_space"):
            space = self._unwrapped.state_space
            return {agent: space for agent in self.possible_agents}
        elif hasattr(self._unwrapped, "state_spaces"):
            return self._unwrapped.state_spaces
        else:
            return {agent: None for agent in self.possible_agents}

    @property
    def observation_spaces(self) -> dict[str, gymnasium.Space]:
        """Observation spaces."""
        return self._unwrapped.observation_spaces

    @property
    def action_spaces(self) -> dict[str, gymnasium.Space]:
        """Action spaces."""
        return self._unwrapped.action_spaces

    def state_space(self, agent: str) -> gymnasium.Space | None:
        """State space.

        See :py:attr:`state_spaces` for more details.

        :param agent: Name of the agent.

        :return: The state space for the specified agent.
        """
        return self.state_spaces[agent]

    def observation_space(self, agent: str) -> gymnasium.Space:
        """Observation space.

        :param agent: Name of the agent.

        :return: The observation space for the specified agent.
        """
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gymnasium.Space:
        """Action space.

        :param agent: Name of the agent.

        :return: The action space for the specified agent.
        """
        return self.action_spaces[agent]
