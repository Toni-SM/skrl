from typing import Any, Mapping, Sequence, Tuple, Union

import gym

import jax
import jax.dlpack
import numpy as np
import torch
import torch.utils.dlpack

from skrl.envs.wrappers.jax.base import MultiAgentEnvWrapper


def _jax2torch(array, device, from_jax=True):
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(array)) if from_jax else torch.tensor(array, device=device)

def _torch2jax(tensor, to_jax=True):
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(tensor.contiguous())) if to_jax else tensor.cpu().numpy()


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

    def step(self, actions: Mapping[str, Union[np.ndarray, jax.Array]]) -> \
        Tuple[Mapping[str, Union[np.ndarray, jax.Array]], Mapping[str, Union[np.ndarray, jax.Array]],
              Mapping[str, Union[np.ndarray, jax.Array]], Mapping[str, Union[np.ndarray, jax.Array]],
              Mapping[str, Any]]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: dict of nd.ndarray or jax.Array

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of dict of nd.ndarray or jax.Array and any other info
        """
        actions = [_jax2torch(actions[uid], self.device, self._jax) for uid in self.possible_agents]

        with torch.no_grad():
            obs_buf, shared_obs_buf, reward_buf, terminated_buf, info, _ = self._env.step(actions)

        obs_buf = _torch2jax(obs_buf, self._jax)
        shared_obs_buf = _torch2jax(shared_obs_buf, self._jax)
        reward_buf = _torch2jax(reward_buf, self._jax)
        terminated_buf = _torch2jax(terminated_buf.to(dtype=torch.int8), self._jax)

        self._obs_buf = {uid: obs_buf[:,i] for i, uid in enumerate(self.possible_agents)}
        self._shared_obs_buf = {uid: shared_obs_buf[:,i] for i, uid in enumerate(self.possible_agents)}
        reward = {uid: reward_buf[:,i].reshape(-1, 1) for i, uid in enumerate(self.possible_agents)}
        terminated = {uid: terminated_buf[:,i].reshape(-1, 1) for i, uid in enumerate(self.possible_agents)}
        truncated = terminated
        info = {"shared_states": self._shared_obs_buf}

        return self._obs_buf, reward, terminated, truncated, info

    def reset(self) -> Tuple[Mapping[str, Union[np.ndarray, jax.Array]], Mapping[str, Any]]:
        """Reset the environment

        :return: Observation, info
        :rtype: tuple of dict of np.ndarray of jax.Array and any other info
        """
        if self._reset_once:
            obs_buf, shared_obs_buf, _ = self._env.reset()

            obs_buf = _torch2jax(obs_buf, self._jax)
            shared_obs_buf = _torch2jax(shared_obs_buf, self._jax)

            self._obs_buf = {uid: obs_buf[:,i] for i, uid in enumerate(self.possible_agents)}
            self._shared_obs_buf = {uid: shared_obs_buf[:,i] for i, uid in enumerate(self.possible_agents)}
            self._reset_once = False
        return self._obs_buf, {"shared_states": self._shared_obs_buf}
