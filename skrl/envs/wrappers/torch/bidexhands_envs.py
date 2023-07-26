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
        self._obs_buf = None
        self._shared_obs_buf = None

        self.possible_agents = [f"agent_{i}" for i in range(self.num_agents)]

    @property
    def agents(self) -> Sequence[str]:
        """Names of all current agents

        These may be changed as an environment progresses (i.e. agents can be added or removed)
        """
        return self.possible_agents

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

    @property
    def shared_observation_spaces(self) -> Mapping[str, gym.Space]:
        """Shared observation spaces
        """
        return {uid: space for uid, space in zip(self.possible_agents, self._env.share_observation_space)}

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
        obs_buf, shared_obs_buf, reward_buf, terminated_buf, info, _ = self._env.step(actions)

        self._obs_buf = {uid: obs_buf[:,i] for i, uid in enumerate(self.possible_agents)}
        self._shared_obs_buf = {uid: shared_obs_buf[:,i] for i, uid in enumerate(self.possible_agents)}
        reward = {uid: reward_buf[:,i].view(-1, 1) for i, uid in enumerate(self.possible_agents)}
        terminated = {uid: terminated_buf[:,i].view(-1, 1) for i, uid in enumerate(self.possible_agents)}
        truncated = {uid: torch.zeros_like(value) for uid, value in terminated.items()}
        info = {"shared_states": self._shared_obs_buf}

        return self._obs_buf, reward, terminated, truncated, info

    def reset(self) -> Tuple[Mapping[str, torch.Tensor], Mapping[str, Any]]:
        """Reset the environment

        :return: Observation, info
        :rtype: tuple of dictionaries of torch.Tensor and any other info
        """
        if self._reset_once:
            obs_buf, shared_obs_buf, _ = self._env.reset()
            self._obs_buf = {uid: obs_buf[:,i] for i, uid in enumerate(self.possible_agents)}
            self._shared_obs_buf = {uid: shared_obs_buf[:,i] for i, uid in enumerate(self.possible_agents)}
            self._reset_once = False
        return self._obs_buf, {"shared_states": self._shared_obs_buf}
