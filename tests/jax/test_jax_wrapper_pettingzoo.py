import pytest

import math
from collections.abc import Mapping
import gymnasium as gym
from pettingzoo.butterfly import pistonball_v6

import jax
import jax.numpy as jnp
import numpy as np

from skrl import config
from skrl.envs.wrappers.jax import PettingZooWrapper, wrap_env


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_env(capsys: pytest.CaptureFixture, backend: str):
    config.jax.backend = backend
    Array = jax.Array if backend == "jax" else np.ndarray

    num_envs = 1
    num_agents = 20
    possible_agents = [f"piston_{i}" for i in range(num_agents)]
    action = {f"piston_{i}": jnp.ones((num_envs, 1)) if backend == "jax" else np.ones((num_envs, 1)) for i in range(num_agents)}

    # load wrap the environment
    original_env = pistonball_v6.parallel_env(n_pistons=num_agents, continuous=True, max_cycles=125)
    env = wrap_env(original_env, "auto")
    assert isinstance(env, PettingZooWrapper)
    env = wrap_env(original_env, "pettingzoo")
    assert isinstance(env, PettingZooWrapper)

    # check properties
    with capsys.disabled():
        pass
    assert isinstance(env.state_spaces, Mapping) and len(env.state_spaces) == num_agents
    assert isinstance(env.observation_spaces, Mapping) and len(env.observation_spaces) == num_agents
    assert isinstance(env.action_spaces, Mapping) and len(env.action_spaces) == num_agents
    for agent in possible_agents:
        assert isinstance(env.state_space(agent), gym.Space) and env.state_space(agent).shape == (560, 880, 3)
        assert isinstance(env.observation_space(agent), gym.Space) and env.observation_space(agent).shape == (457, 120, 3)
        assert isinstance(env.action_space(agent), gym.Space) and env.action_space(agent).shape == (1,)
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
        for agent in possible_agents:
            assert isinstance(observation[agent], Array) and observation[agent].shape == (num_envs, math.prod((457, 120, 3)))
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step(action)
            state = env.state()
            env.render()
            assert isinstance(observation, Mapping)
            assert isinstance(reward, Mapping)
            assert isinstance(terminated, Mapping)
            assert isinstance(truncated, Mapping)
            assert isinstance(info, Mapping)
            for agent in possible_agents:
                assert isinstance(observation[agent], Array) and observation[agent].shape == (num_envs, math.prod((457, 120, 3)))
                assert isinstance(reward[agent], Array) and reward[agent].shape == (num_envs, 1)
                assert isinstance(terminated[agent], Array) and terminated[agent].shape == (num_envs, 1)
                assert isinstance(truncated[agent], Array) and truncated[agent].shape == (num_envs, 1)
            assert isinstance(state, Array) and state.shape == (num_envs, math.prod((560, 880, 3)))

    env.close()
