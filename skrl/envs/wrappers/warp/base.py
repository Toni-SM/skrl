from typing import Any, Mapping, Sequence, Tuple, Union

from abc import ABC, abstractmethod
import gymnasium

import warp as wp

from skrl import config


class Wrapper(ABC):
    def __init__(self, env: Any) -> None:
        """Base wrapper class for RL environments.

        :param env: The environment instance to wrap.
        """
        self._env = env
        try:
            self._unwrapped = self._env.unwrapped
        except:
            self._unwrapped = env

        # device
        if hasattr(self._unwrapped, "device"):
            self._device = config.warp.parse_device(self._unwrapped.device)
        else:
            self._device = config.warp.parse_device(None)

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
    def reset(self) -> Tuple[wp.array, Any]:
        """Reset the environment.

        :return: Observation, info.
        """
        pass

    @abstractmethod
    def step(self, actions: wp.array) -> Tuple[wp.array, wp.array, wp.array, wp.array, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        pass

    @abstractmethod
    def state(self) -> Union[wp.array, None]:
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
    def device(self) -> wp.context.Device:
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
    def state_space(self) -> Union[gymnasium.Space, None]:
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
    pass
