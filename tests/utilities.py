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


def _sample_flattened_space(*, space, num_envs, device, ml_framework):
    if ml_framework == "torch":
        from skrl.utils.spaces.torch import flatten_tensorized_space, sample_space
    elif ml_framework == "jax":
        from skrl.utils.spaces.jax import flatten_tensorized_space, sample_space

    return flatten_tensorized_space(sample_space(space, batch_size=num_envs, backend="native", device=device))


def _sample_flattened_spaces(*, spaces, num_envs, device, ml_framework):
    if ml_framework == "torch":
        from skrl.utils.spaces.torch import flatten_tensorized_space, sample_space
    elif ml_framework == "jax":
        from skrl.utils.spaces.jax import flatten_tensorized_space, sample_space

    return {
        uid: flatten_tensorized_space(sample_space(space, batch_size=num_envs, backend="native", device=device))
        for uid, space in spaces.items()
    }


def _check_flattened_space(*, sample, space, num_envs, ml_framework):
    if ml_framework == "torch":
        from skrl.utils.spaces.torch import compute_space_size
    elif ml_framework == "jax":
        from skrl.utils.spaces.jax import compute_space_size

    space_size = compute_space_size(space, occupied_size=True)
    assert sample.shape[0] == num_envs, f"Space dim 0 mismatch: expected {num_envs}, got {sample.shape[0]}"
    assert sample.shape[1] == space_size, f"Space dim 1 mismatch: expected {space_size}, got {sample.shape[1]}"


def _check_flattened_spaces(*, sample, spaces, num_envs, ml_framework):
    if ml_framework == "torch":
        from skrl.utils.spaces.torch import compute_space_size
    elif ml_framework == "jax":
        from skrl.utils.spaces.jax import compute_space_size

    for uid, space in spaces.items():
        if space is None:
            continue
        space_size = compute_space_size(space, occupied_size=True)
        assert (
            sample[uid].shape[0] == num_envs
        ), f"Space dim 0 mismatch: expected {num_envs}, got {sample[uid].shape[0]}"
        assert (
            sample[uid].shape[1] == space_size
        ), f"Space dim 1 mismatch: expected {space_size}, got {sample[uid].shape[1]}"


class SingleAgentEnv:
    def __init__(
        self, *, observation_space, state_space, action_space, num_envs, device, ml_framework, probability=0.05
    ):
        assert ml_framework in ["torch", "jax"]
        self._ml_framework = ml_framework
        self._probability = probability

        # Wrapper properties
        self.observation_space = observation_space
        self.state_space = state_space
        self.action_space = action_space
        self.num_envs = num_envs
        self.num_agents = 1
        if self._ml_framework == "torch":
            self.device = config.torch.parse_device(device)
        elif self._ml_framework == "jax":
            self.device = config.jax.parse_device(device)

    def _tensorize(self, x, dtype):
        if self._ml_framework == "torch":
            import torch

            dtype = {bool: torch.bool, int: torch.int, float: torch.float}[dtype]
            return torch.tensor(x, device=self.device, dtype=dtype).view(self.num_envs, -1)
        elif self._ml_framework == "jax":
            import jax

            dtype = {bool: np.int8, int: np.int32, float: np.float32}[dtype]
            return jax.device_put(np.array(x, dtype=dtype).reshape(self.num_envs, -1), device=self.device)

    # Wrapper methods

    def step(self, actions):
        _check_flattened_space(
            sample=actions, space=self.action_space, num_envs=self.num_envs, ml_framework=self._ml_framework
        )

        observations = _sample_flattened_space(
            space=self.observation_space, num_envs=self.num_envs, device=self.device, ml_framework=self._ml_framework
        )
        if self.num_envs == 1:
            rewards = random.random()
            terminated = random.random() < self._probability
            truncated = random.random() < self._probability
        else:
            rewards = np.random.random((self.num_envs,))
            terminated = np.random.random((self.num_envs,)) < self._probability
            truncated = np.random.random((self.num_envs,)) < self._probability

        return (
            observations,
            self._tensorize(rewards, float),
            self._tensorize(terminated, bool),
            self._tensorize(truncated, bool),
            {},
        )

    def state(self):
        if self.state_space is None:
            return None
        return _sample_flattened_space(
            space=self.state_space, num_envs=self.num_envs, device=self.device, ml_framework=self._ml_framework
        )

    def reset(self):
        observations = _sample_flattened_space(
            space=self.observation_space, num_envs=self.num_envs, device=self.device, ml_framework=self._ml_framework
        )
        return observations, {}

    def render(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


class MultiAgentEnv:
    def __init__(
        self, *, observation_spaces, state_spaces, action_spaces, num_envs, device, ml_framework, probability=0.05
    ):
        assert ml_framework in ["torch", "jax"]
        self._ml_framework = ml_framework
        self._probability = probability

        # Wrapper properties
        self.observation_spaces = observation_spaces
        self.state_spaces = state_spaces
        self.action_spaces = action_spaces
        self.max_num_agents = len(observation_spaces)
        self.num_envs = num_envs
        self.num_agents = self.max_num_agents
        self.possible_agents = sorted(self.observation_spaces.keys())
        self.agents = self.possible_agents
        if self._ml_framework == "torch":
            self.device = config.torch.parse_device(device)
        elif self._ml_framework == "jax":
            self.device = config.jax.parse_device(device)

    def _tensorize(self, x, dtype):
        if self._ml_framework == "torch":
            import torch

            dtype = {bool: torch.bool, int: torch.int, float: torch.float}[dtype]
            return {
                uid: torch.tensor(x, device=self.device, dtype=dtype).view(self.num_envs, -1)
                for uid in self.possible_agents
            }
        elif self._ml_framework == "jax":
            import jax

            dtype = {bool: np.int8, int: np.int32, float: np.float32}[dtype]
            return {
                uid: jax.device_put(np.array(x, dtype=dtype).reshape(self.num_envs, -1), device=self.device)
                for uid in self.possible_agents
            }

    # Wrapper methods

    def state_space(self, agent):
        return self.state_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def step(self, actions):
        _check_flattened_spaces(
            sample=actions, spaces=self.action_spaces, num_envs=self.num_envs, ml_framework=self._ml_framework
        )

        observations = _sample_flattened_spaces(
            spaces=self.observation_spaces, num_envs=self.num_envs, device=self.device, ml_framework=self._ml_framework
        )
        if self.num_envs == 1:
            rewards = random.random()
            terminated = random.random() < self._probability
            truncated = random.random() < self._probability
        else:
            rewards = np.random.random((self.num_envs,))
            terminated = np.random.random((self.num_envs,)) < self._probability
            truncated = np.random.random((self.num_envs,)) < self._probability

        return (
            observations,
            self._tensorize(rewards, float),
            self._tensorize(terminated, bool),
            self._tensorize(truncated, bool),
            {},
        )

    def state(self):
        if self.state_spaces is None:
            return None
        return _sample_flattened_spaces(
            spaces=self.state_spaces, num_envs=self.num_envs, device=self.device, ml_framework=self._ml_framework
        )

    def reset(self):
        observations = _sample_flattened_spaces(
            spaces=self.observation_spaces, num_envs=self.num_envs, device=self.device, ml_framework=self._ml_framework
        )
        return observations, {}

    def render(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


class _AgentMock:
    def __init__(self, *, observation_space, state_space, action_space, num_envs, device, ml_framework, **kwargs):
        assert ml_framework in ["torch", "jax"]
        self._ml_framework = ml_framework

        # Agent properties
        self.observation_space = observation_space
        self.state_space = state_space
        self.action_space = action_space
        self.num_envs = num_envs
        if self._ml_framework == "torch":
            self.device = config.torch.parse_device(device)
        elif self._ml_framework == "jax":
            self.device = config.jax.parse_device(device)

        self.memory = None
        self.models = {}

    def init(self, *, trainer_cfg):
        pass

    def enable_training_mode(self, enabled):
        pass

    def enable_models_training_mode(self, enabled):
        pass

    def track_data(self, key, value):
        pass

    def act(self, observations, states, *, timestep, timesteps):
        _check_flattened_space(
            sample=observations, space=self.observation_space, num_envs=self.num_envs, ml_framework=self._ml_framework
        )
        if states is not None:
            _check_flattened_space(
                sample=states, space=self.state_space, num_envs=self.num_envs, ml_framework=self._ml_framework
            )
        actions = _sample_flattened_space(
            space=self.action_space, num_envs=self.num_envs, device=self.device, ml_framework=self._ml_framework
        )
        return actions, {"mean_actions": actions}

    def record_transition(
        self,
        *,
        observations,
        states,
        actions,
        rewards,
        next_observations,
        next_states,
        terminated,
        truncated,
        infos,
        timestep,
        timesteps,
    ):
        _check_flattened_space(
            sample=observations, space=self.observation_space, num_envs=self.num_envs, ml_framework=self._ml_framework
        )
        _check_flattened_space(
            sample=next_observations,
            space=self.observation_space,
            num_envs=self.num_envs,
            ml_framework=self._ml_framework,
        )
        if states is not None:
            _check_flattened_space(
                sample=states, space=self.state_space, num_envs=self.num_envs, ml_framework=self._ml_framework
            )
        if next_states is not None:
            _check_flattened_space(
                sample=next_states, space=self.state_space, num_envs=self.num_envs, ml_framework=self._ml_framework
            )
        _check_flattened_space(
            sample=actions, space=self.action_space, num_envs=self.num_envs, ml_framework=self._ml_framework
        )

    def pre_interaction(self, *, timestep, timesteps):
        pass

    def post_interaction(self, *, timestep, timesteps):
        pass


class AgentMock(_AgentMock):
    pass


class _MultiAgentMock:
    def __init__(
        self,
        *,
        possible_agents,
        observation_spaces,
        state_spaces,
        action_spaces,
        num_envs,
        device,
        ml_framework,
        **kwargs,
    ):
        assert ml_framework in ["torch", "jax"]
        self._ml_framework = ml_framework

        # Multi-agent properties
        self.possible_agents = possible_agents
        self.observation_spaces = observation_spaces
        self.state_spaces = state_spaces
        self.action_spaces = action_spaces
        self.num_envs = num_envs
        self.num_agents = len(self.possible_agents)
        if self._ml_framework == "torch":
            self.device = config.torch.parse_device(device)
        elif self._ml_framework == "jax":
            self.device = config.jax.parse_device(device)

        self.memories = {}
        self.models = {}

    def init(self, *, trainer_cfg):
        pass

    def enable_training_mode(self, enabled):
        pass

    def enable_models_training_mode(self, enabled):
        pass

    def track_data(self, key, value):
        pass

    def act(self, observations, states, *, timestep, timesteps):
        _check_flattened_spaces(
            sample=observations, spaces=self.observation_spaces, num_envs=self.num_envs, ml_framework=self._ml_framework
        )
        _check_flattened_spaces(
            sample=states, spaces=self.state_spaces, num_envs=self.num_envs, ml_framework=self._ml_framework
        )
        actions = _sample_flattened_spaces(
            spaces=self.action_spaces, num_envs=self.num_envs, device=self.device, ml_framework=self._ml_framework
        )
        outputs = {uid: {"mean_actions": actions[uid]} for uid in actions}
        return actions, outputs

    def record_transition(
        self,
        *,
        observations,
        states,
        actions,
        rewards,
        next_observations,
        next_states,
        terminated,
        truncated,
        infos,
        timestep,
        timesteps,
    ):
        _check_flattened_spaces(
            sample=observations, spaces=self.observation_spaces, num_envs=self.num_envs, ml_framework=self._ml_framework
        )
        _check_flattened_spaces(
            sample=next_observations,
            spaces=self.observation_spaces,
            num_envs=self.num_envs,
            ml_framework=self._ml_framework,
        )
        _check_flattened_spaces(
            sample=states, spaces=self.state_spaces, num_envs=self.num_envs, ml_framework=self._ml_framework
        )
        _check_flattened_spaces(
            sample=next_states, spaces=self.state_spaces, num_envs=self.num_envs, ml_framework=self._ml_framework
        )
        _check_flattened_spaces(
            sample=actions, spaces=self.action_spaces, num_envs=self.num_envs, ml_framework=self._ml_framework
        )

    def pre_interaction(self, *, timestep, timesteps):
        pass

    def post_interaction(self, *, timestep, timesteps):
        pass


class MultiAgentMock(_MultiAgentMock):
    pass
