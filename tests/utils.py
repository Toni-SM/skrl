import random
import gymnasium

import numpy as np


class BaseEnv(gymnasium.Env):
    def __init__(self, observation_space, action_space, num_envs, device):
        self.device = device
        self.num_envs = num_envs
        self.action_space = action_space
        self.observation_space = observation_space

    def _sample_observation(self):
        raise NotImplementedError

    def step(self, actions):
        if self.num_envs == 1:
            rewards = random.random()
            terminated = random.random() > 0.95
            truncated = random.random() > 0.95
        else:
            rewards = np.random.random((self.num_envs,))
            terminated = np.random.random((self.num_envs,)) > 0.95
            truncated = np.random.random((self.num_envs,)) > 0.95

        return self._sample_observation(), rewards, terminated, truncated, {}

    def reset(self):
        return self._sample_observation(), {}

    def render(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass
