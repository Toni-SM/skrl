import os
import random
import gymnasium

import numpy as np


def is_device_available(device, *, backend) -> bool:
    if backend == "torch":
        import torch

        try:
            torch.zeros((1,), device=device)
        except Exception as e:
            return False
    else:
        raise ValueError(f"Invalid backend: {backend}")
    return True


def get_test_mixed_precision(default):
    value = os.environ.get("SKRL_TEST_MIXED_PRECISION")
    if value is None:
        return False
    if value.lower() in ["true", "1", "y", "yes"]:
        return default
    if value.lower() in ["false", "0", "n", "no"]:
        return False
    raise ValueError(f"Invalid value for environment variable SKRL_TEST_MIXED_PRECISION: {value}")


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
