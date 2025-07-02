import os
import random

import numpy as np

from skrl import config


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


class SingleAgentEnv:
    def __init__(self, *, observation_space, state_space, action_space, num_envs, device, ml_framework):
        assert ml_framework in ["torch", "jax"]
        self.ml_framework = ml_framework

        # Wrapper properties
        self.observation_space = observation_space
        self.state_space = state_space
        self.action_space = action_space
        self.num_envs = num_envs
        self.num_agents = 1
        if self.ml_framework == "torch":
            self.device = config.torch.parse_device(device)
        elif self.ml_framework == "jax":
            self.device = config.jax.parse_device(device)

    def _tensorize(self, x, dtype):
        if self.ml_framework == "torch":
            import torch

            dtype = {bool: torch.bool, int: torch.int, float: torch.float}[dtype]
            return torch.tensor(x, device=self.device, dtype=dtype).view(self.num_envs, -1)
        elif self.ml_framework == "jax":
            import jax

            dtype = {bool: np.int8, int: np.int32, float: np.float32}[dtype]
            return jax.device_put(np.array(x, dtype=dtype), device=self.device)

    def _sample_observation(self):
        if self.ml_framework == "torch":
            from skrl.utils.spaces.torch import flatten_tensorized_space, sample_space
        elif self.ml_framework == "jax":
            from skrl.utils.spaces.jax import flatten_tensorized_space, sample_space
        return flatten_tensorized_space(
            sample_space(self.observation_space, batch_size=self.num_envs, backend="native", device=self.device)
        )

    def _sample_state(self):
        if self.ml_framework == "torch":
            from skrl.utils.spaces.torch import flatten_tensorized_space, sample_space
        elif self.ml_framework == "jax":
            from skrl.utils.spaces.jax import flatten_tensorized_space, sample_space

        return flatten_tensorized_space(
            sample_space(self.state_space, batch_size=self.num_envs, backend="native", device=self.device)
        )

    # Wrapper methods

    def step(self, actions):
        if self.num_envs == 1:
            rewards = random.random()
            terminated = random.random() > 0.95
            truncated = random.random() > 0.95
        else:
            rewards = np.random.random((self.num_envs,))
            terminated = np.random.random((self.num_envs,)) > 0.95
            truncated = np.random.random((self.num_envs,)) > 0.95

        return (
            self._sample_observation(),
            self._tensorize(rewards, float),
            self._tensorize(terminated, bool),
            self._tensorize(truncated, bool),
            {},
        )

    def state(self):
        return None if self.state_space is None else self._sample_state()

    def reset(self):
        return self._sample_observation(), {}

    def render(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass
