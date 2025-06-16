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


def is_running_on_github_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") is not None


def get_test_mixed_precision(default):
    value = os.environ.get("SKRL_TEST_MIXED_PRECISION")
    if value is None:
        return False
    if value.lower() in ["true", "1", "y", "yes"]:
        return default
    if value.lower() in ["false", "0", "n", "no"]:
        return False
    raise ValueError(f"Invalid value for environment variable SKRL_TEST_MIXED_PRECISION: {value}")


def check_config_keys(config, default_config):
    for k, v in config.items():
        assert k in default_config, f"Key '{k}' not in default config"
        if isinstance(v, dict) and k == "experiment":
            check_config_keys(config[k], default_config[k])
    for k, v in default_config.items():
        assert k in config, f"Key '{k}' not in config"
        if isinstance(v, dict) and k == "experiment":
            check_config_keys(config[k], default_config[k])


class BaseEnv(gymnasium.Env):
    def __init__(self, *, observation_space, state_space, action_space, num_envs, device):
        self.device = device
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.state_space = state_space
        self.action_space = action_space

    def _sample_observation(self):
        raise NotImplementedError

    def _sample_state(self):
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

    def state(self):
        return None if self.state_space is None else self._sample_state()

    def reset(self):
        return self._sample_observation(), {}

    def render(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass
