from typing import Any, Mapping, Sequence, Tuple

import gym

import torch


class Wrapper(object):
    def __init__(self, env: Any) -> None:
        """Base wrapper class for RL environments

        :param env: The environment to wrap
        :type env: Any supported RL environment
        """
        self._env = env

        # device (faster than @property)
        if hasattr(self._env, "device"):
            self.device = torch.device(self._env.device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :raises NotImplementedError: Not implemented

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        raise NotImplementedError

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :raises NotImplementedError: Not implemented

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
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
        self._env = env
        try:
            self._unwrapped = self._env.unwrapped
        except:
            self._unwrapped = env

        # device (faster than @property)
        if hasattr(self._unwrapped, "device"):
            self.device = torch.device(self._unwrapped.device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    def reset(self) -> Tuple[Mapping[str, torch.Tensor], Mapping[str, Any]]:
        """Reset the environment

        :raises NotImplementedError: Not implemented

        :return: Observation, info
        :rtype: tuple of dictionaries of torch.Tensor and any other info
        """
        raise NotImplementedError

    def step(self, actions: Mapping[str, torch.Tensor]) -> \
        Tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor],
              Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], Mapping[str, Any]]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: dictionary of torch.Tensor

        :raises NotImplementedError: Not implemented

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of dictionaries of torch.Tensor and any other info
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
