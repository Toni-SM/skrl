import pytest

from collections.abc import Mapping
import gymnasium as gym

import torch

from skrl.envs.loaders.torch import load_playground_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.envs.wrappers.torch.playground_envs import PlaygroundWrapper

from ....utilities import is_device_available, is_running_on_github_actions


@pytest.mark.parametrize("task_name", ["CartpoleBalance", "LeapCubeReorient"])
def test_env(capsys: pytest.CaptureFixture, task_name: str):
    num_envs = 10
    num_observations = 5 if task_name == "CartpoleBalance" else 57
    num_states = 0 if task_name == "CartpoleBalance" else 128
    num_actions = 1 if task_name == "CartpoleBalance" else 16
    action = torch.ones((num_envs, num_actions))

    # check wrapper definition
    with pytest.raises(AttributeError):
        assert isinstance(wrap_env(None, "playground"), PlaygroundWrapper)

    # load wrap the environment
    try:
        import mujoco_playground
    except ImportError as e:
        if is_running_on_github_actions():
            raise e
        else:
            pytest.skip(f"Unable to import MuJoCo Playground environment: {e}")

    # load and wrap the environment
    cfg_overrides = None if is_device_available("cuda", backend="torch") else {"impl": "jax"}  # warp impl requires GPU
    original_env = load_playground_env(task_name=task_name, num_envs=num_envs, cfg_overrides=cfg_overrides)
    env = wrap_env(original_env, "auto")
    assert isinstance(env, PlaygroundWrapper)
    env = wrap_env(original_env, "playground")
    assert isinstance(env, PlaygroundWrapper)

    # check properties
    if num_states:
        assert isinstance(env.state_space, gym.Space) and env.state_space.shape == (num_states,)
    else:
        assert env.state_space is None
    assert isinstance(env.observation_space, gym.Space) and env.observation_space.shape == (num_observations,)
    assert isinstance(env.action_space, gym.Space) and env.action_space.shape == (num_actions,)
    assert isinstance(env.num_envs, int) and env.num_envs == num_envs
    assert isinstance(env.num_agents, int) and env.num_agents == 1
    assert isinstance(env.device, torch.device)
    # check internal properties
    assert env._env is original_env
    assert env._unwrapped is original_env.unwrapped
    # check methods
    for _ in range(2):
        observation, info = env.reset()
        state = env.state()
        observation, info = env.reset()  # edge case: parallel environments are autoreset
        state = env.state()
        assert isinstance(observation, torch.Tensor) and observation.shape == torch.Size([num_envs, num_observations])
        assert isinstance(info, Mapping)
        if num_states:
            assert isinstance(state, torch.Tensor) and state.shape == (num_envs, num_states)
        else:
            assert state is None
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step(action)
            state = env.state()
            env.render()
            assert isinstance(observation, torch.Tensor) and observation.shape == torch.Size(
                [num_envs, num_observations]
            )
            assert isinstance(reward, torch.Tensor) and reward.shape == torch.Size([num_envs, 1])
            assert isinstance(terminated, torch.Tensor) and terminated.shape == torch.Size([num_envs, 1])
            assert isinstance(truncated, torch.Tensor) and truncated.shape == torch.Size([num_envs, 1])
            assert isinstance(info, Mapping)
            if num_states:
                assert isinstance(state, torch.Tensor) and state.shape == (num_envs, num_states)
            else:
                assert state is None

    env.close()
