from typing import Any, Mapping, Sequence, Tuple

import gym

import torch

from skrl.envs.wrappers.torch.base import MultiAgentEnvWrapper


class BiDexHandsWrapper(MultiAgentEnvWrapper):
    def __init__(self, env: Any) -> None:
        """Bi-DexHands wrapper

        :param env: The environment to wrap
        :type env: Any supported Bi-DexHands environment
        """
        super().__init__(env)

        self._reset_once = True
        self._states = None
        self._observations = None
        self._info = {}

    @property
    def agents(self) -> Sequence[str]:
        """Names of all current agents

        These may be changed as an environment progresses (i.e. agents can be added or removed)
        """
        return self.possible_agents

    @property
    def possible_agents(self) -> Sequence[str]:
        """Names of all possible agents the environment could generate

        These can not be changed as an environment progresses
        """
        return [f"agent_{i}" for i in range(self.num_agents)]

    @property
    def state_spaces(self) -> Mapping[str, gym.Space]:
        """State spaces

        Since the state space is a global view of the environment (and therefore the same for all the agents),
        this property returns a dictionary (for consistency with the other space-related properties) with the same
        space for all the agents
        """
        return {uid: space for uid, space in zip(self.possible_agents, self._env.share_observation_space)}

    @property
    def observation_spaces(self) -> Mapping[str, gym.Space]:
        """Observation spaces
        """
        return {uid: space for uid, space in zip(self.possible_agents, self._env.observation_space)}

    @property
    def action_spaces(self) -> Mapping[str, gym.Space]:
        """Action spaces
        """
        return {uid: space for uid, space in zip(self.possible_agents, self._env.action_space)}

    def step(self, actions: Mapping[str, torch.Tensor]) -> \
        Tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor],
              Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], Mapping[str, Any]]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: dictionary of torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of dictionaries torch.Tensor and any other info
        """
        actions = [actions[uid] for uid in self.possible_agents]
        observations, states, rewards, terminated, _, _ = self._env.step(actions)

        self._states = states[:, 0]
        self._observations = {uid: observations[:,i] for i, uid in enumerate(self.possible_agents)}
        rewards = {uid: rewards[:,i].view(-1, 1) for i, uid in enumerate(self.possible_agents)}
        terminated = {uid: terminated[:,i].view(-1, 1) for i, uid in enumerate(self.possible_agents)}
        truncated = {uid: torch.zeros_like(value) for uid, value in terminated.items()}

        return self._observations, rewards, terminated, truncated, self._info

    def state(self) -> torch.Tensor:
        """Get the environment state

        :return: State
        :rtype: torch.Tensor
        """
        return self._states

    def reset(self) -> Tuple[Mapping[str, torch.Tensor], Mapping[str, Any]]:
        """Reset the environment

        :return: Observation, info
        :rtype: tuple of dictionaries of torch.Tensor and any other info
        """
        if self._reset_once:
            observations, states, _ = self._env.reset()
            self._states = states[:, 0]
            self._observations = {uid: observations[:,i] for i, uid in enumerate(self.possible_agents)}
            self._reset_once = False
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        return None

    def close(self) -> None:
        """Close the environment
        """
        pass
