import pytest

from collections.abc import Mapping
import gymnasium as gym

import jax
import jax.numpy as jnp
import numpy as np
import torch

from skrl import config
from skrl.envs.wrappers.jax import IsaacLabMultiAgentWrapper, IsaacLabWrapper, wrap_env


class IsaacLabEnv(gym.Env):
    def __init__(self, num_states):
        self.num_actions = 1
        self.num_observations = 4
        self.num_states = num_states
        self.num_envs = 10
        self.extras = {}
        self.device = "cpu"

        self._configure_gym_env_spaces()

    # https://github.com/isaac-sim/IsaacLab/blob/main/source/extensions/omni.isaac.lab/omni/isaac/lab/envs/direct_rl_env.py
    def _configure_gym_env_spaces(self):
        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_observations,)
        )
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_actions,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # optional state space for asymmetric actor-critic architectures
        if self.num_states > 0:
            self.single_observation_space["critic"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,))
            self.state_space = gym.vector.utils.batch_space(self.single_observation_space["critic"], self.num_envs)

    def reset(self, seed=None, options=None):
        observations = {"policy": torch.ones((self.num_envs, self.num_observations), device=self.device)}
        return observations, self.extras

    def step(self, action):
        assert action.clone().shape == torch.Size([self.num_envs, 1])
        observations = {
            "policy": torch.ones((self.num_envs, self.num_observations), device=self.device, dtype=torch.float32)
        }
        rewards = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        return observations, rewards, terminated, truncated, self.extras

    def render(self, recompute=False):
        return None

    def close(self):
        pass


class IsaacLabMultiAgentEnv:
    def __init__(self, num_states):
        self.possible_agents = [f"agent_{i}" for i in range(3)]
        self.agents = self.possible_agents
        self.num_actions = {f"agent_{i}": i + 10 for i in range(len(self.possible_agents))}
        self.num_observations = {f"agent_{i}": i + 20 for i in range(len(self.possible_agents))}
        self.num_states = num_states
        self.num_envs = 10
        self.extras = {agent: {} for agent in self.possible_agents}
        self.device = "cpu"

        self._configure_env_spaces()

    # https://github.com/isaac-sim/IsaacLab/blob/main/source/extensions/omni.isaac.lab/omni/isaac/lab/envs/direct_marl_env.py
    def _configure_env_spaces(self):
        # set up observation and action spaces
        self.observation_spaces = {
            agent: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_observations[agent],))
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_actions[agent],))
            for agent in self.possible_agents
        }
        # set up state space
        if not self.num_states:
            self.state_space = None
        if self.num_states < 0:
            self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(sum(self.num_observations.values()),))
        else:
            self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,))

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None, options=None):
        observations = {
            agent: torch.ones((self.num_envs, self.num_observations[agent]), device=self.device)
            for agent in self.possible_agents
        }
        return observations, self.extras

    def step(self, action):
        for agent in self.possible_agents:
            assert action[agent].clone().shape == torch.Size([self.num_envs, self.num_actions[agent]])
        observations = {
            agent: torch.ones((self.num_envs, self.num_observations[agent]), device=self.device)
            for agent in self.possible_agents
        }
        rewards = {
            agent: torch.zeros(self.num_envs, device=self.device, dtype=torch.float32) for agent in self.possible_agents
        }
        terminated = {
            agent: torch.zeros(self.num_envs, device=self.device, dtype=torch.bool) for agent in self.possible_agents
        }
        truncated = {agent: torch.zeros_like(terminated[agent]) for agent in self.possible_agents}
        return observations, rewards, terminated, truncated, self.extras

    def state(self):
        if not self.num_states:
            return None
        return torch.ones((self.num_envs, self.num_states), device=self.device)

    def render(self, recompute=False):
        return None

    def close(self):
        pass


@pytest.mark.parametrize("backend", ["jax", "numpy"])
@pytest.mark.parametrize("num_states", [0, 5])
def test_env(capsys: pytest.CaptureFixture, backend: str, num_states: int):
    config.jax.backend = backend
    Array = jax.Array if backend == "jax" else np.ndarray

    num_envs = 10
    action = jnp.ones((num_envs, 1)) if backend == "jax" else np.ones((num_envs, 1))

    # load wrap the environment
    original_env = IsaacLabEnv(num_states)
    env = wrap_env(original_env, "auto")
    # TODO: assert isinstance(env, IsaacLabWrapper)
    env = wrap_env(original_env, "isaaclab")
    assert isinstance(env, IsaacLabWrapper)

    # check properties
    if num_states:
        assert isinstance(env.state_space, gym.Space) and env.state_space.shape == (num_states,)
    else:
        assert env.state_space is None or env.state_space.shape == (num_states,)
    assert isinstance(env.observation_space, gym.Space) and env.observation_space.shape == (4,)
    assert isinstance(env.action_space, gym.Space) and env.action_space.shape == (1,)
    assert isinstance(env.num_envs, int) and env.num_envs == num_envs
    assert isinstance(env.num_agents, int) and env.num_agents == 1
    assert isinstance(env.device, jax.Device)
    # check internal properties
    assert env._env is original_env
    assert env._unwrapped is original_env.unwrapped
    # check methods
    for _ in range(2):
        observation, info = env.reset()
        observation, info = env.reset()  # edge case: parallel environments are autoreset
        assert isinstance(observation, Array) and observation.shape == (num_envs, 4)
        assert isinstance(info, Mapping)
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            assert isinstance(observation, Array) and observation.shape == (num_envs, 4)
            assert isinstance(reward, Array) and reward.shape == (num_envs, 1)
            assert isinstance(terminated, Array) and terminated.shape == (num_envs, 1)
            assert isinstance(truncated, Array) and truncated.shape == (num_envs, 1)
            assert isinstance(info, Mapping)

    env.close()


@pytest.mark.parametrize("backend", ["jax", "numpy"])
@pytest.mark.parametrize("num_states", [0, 5])
def test_multi_agent_env(capsys: pytest.CaptureFixture, backend: str, num_states: int):
    config.jax.backend = backend
    Array = jax.Array if backend == "jax" else np.ndarray

    num_envs = 10
    num_agents = 3
    possible_agents = [f"agent_{i}" for i in range(num_agents)]
    action = {
        f"agent_{i}": jnp.ones((num_envs, i + 10)) if backend == "jax" else np.ones((num_envs, i + 10))
        for i in range(num_agents)
    }

    # load wrap the environment
    original_env = IsaacLabMultiAgentEnv(num_states)
    # TODO: env = wrap_env(original_env, "auto")
    # TODO: assert isinstance(env, IsaacLabMultiAgentWrapper)
    env = wrap_env(original_env, "isaaclab-multi-agent")
    assert isinstance(env, IsaacLabMultiAgentWrapper)

    # check properties
    assert isinstance(env.state_spaces, Mapping) and len(env.state_spaces) == num_agents
    assert isinstance(env.observation_spaces, Mapping) and len(env.observation_spaces) == num_agents
    assert isinstance(env.action_spaces, Mapping) and len(env.action_spaces) == num_agents
    for i, agent in enumerate(possible_agents):
        assert isinstance(env.state_space(agent), gym.Space) and env.state_space(agent).shape == (num_states,)
        assert isinstance(env.observation_space(agent), gym.Space) and env.observation_space(agent).shape == (i + 20,)
        assert isinstance(env.action_space(agent), gym.Space) and env.action_space(agent).shape == (i + 10,)
    assert isinstance(env.possible_agents, list) and sorted(env.possible_agents) == sorted(possible_agents)
    assert isinstance(env.num_envs, int) and env.num_envs == num_envs
    assert isinstance(env.num_agents, int) and env.num_agents == num_agents
    assert isinstance(env.max_num_agents, int) and env.max_num_agents == num_agents
    assert isinstance(env.device, jax.Device)
    # check internal properties
    assert env._env is original_env
    assert env._unwrapped is original_env.unwrapped
    # check methods
    for _ in range(2):
        observation, info = env.reset()
        assert isinstance(observation, Mapping)
        assert isinstance(info, Mapping)
        for i, agent in enumerate(possible_agents):
            assert isinstance(observation[agent], Array) and observation[agent].shape == (num_envs, i + 20)
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step(action)
            state = env.state()
            env.render()
            assert isinstance(observation, Mapping)
            assert isinstance(reward, Mapping)
            assert isinstance(terminated, Mapping)
            assert isinstance(truncated, Mapping)
            assert isinstance(info, Mapping)
            for i, agent in enumerate(possible_agents):
                assert isinstance(observation[agent], Array) and observation[agent].shape == (num_envs, i + 20)
                assert isinstance(reward[agent], Array) and reward[agent].shape == (num_envs, 1)
                assert isinstance(terminated[agent], Array) and terminated[agent].shape == (num_envs, 1)
                assert isinstance(truncated[agent], Array) and truncated[agent].shape == (num_envs, 1)
            if num_states:
                assert isinstance(state, Array) and state.shape == (num_envs, num_states)
            else:
                assert state is None

    env.close()
