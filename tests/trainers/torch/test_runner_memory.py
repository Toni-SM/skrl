import pytest

import copy
import gymnasium

import numpy as np

from skrl.agents.torch.ppo import PPO_CFG
from skrl.utils.runner.torch import Runner

from ...utilities import SingleAgentEnv


def parameter_arrays(model):
    return [value.detach().cpu().numpy().copy() for value in model.parameters()]


@pytest.mark.parametrize("memory_size,rollouts", [(-1, None), (-1, 4), (16, None)])
def test_runner_memory_uses_effective_rollouts(memory_size, rollouts):
    expected_rollouts = PPO_CFG().rollouts if rollouts is None else rollouts
    expected_memory_size = expected_rollouts if memory_size < 0 else memory_size
    env = SingleAgentEnv(
        observation_space=gymnasium.spaces.Box(low=-1, high=1, shape=(2,)),
        state_space=None,
        action_space=gymnasium.spaces.Box(low=-1, high=1, shape=(1,)),
        num_envs=2,
        device="cpu",
        ml_framework="torch",
        probability=0,
    )
    network = [{"name": "net", "input": "OBSERVATIONS", "layers": [8], "activations": "tanh"}]
    cfg = {
        "seed": 17,
        "models": {
            "separate": True,
            "policy": {"class": "GaussianMixin", "clip_actions": True, "network": network, "output": "ACTIONS"},
            "value": {"class": "DeterministicMixin", "network": network, "output": "ONE"},
        },
        "memory": {"class": "RandomMemory", "memory_size": memory_size},
        "agent": {
            "class": "PPO",
            "learning_epochs": 1,
            "mini_batches": 1,
            "learning_rate": 0.001,
            "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        },
        "trainer": {
            "class": "SequentialTrainer",
            "timesteps": 2 * expected_rollouts,
            "headless": True,
            "disable_progressbar": True,
            "close_environment_at_exit": False,
        },
    }
    if rollouts is not None:
        cfg["agent"]["rollouts"] = rollouts
    original_cfg = copy.deepcopy(cfg)
    runner = Runner(env, cfg)
    assert runner.agent.cfg.rollouts == expected_rollouts
    assert runner.agent.memory.memory_size == expected_memory_size
    before = parameter_arrays(runner.agent.policy)
    runner.run("train")
    after = parameter_arrays(runner.agent.policy)
    assert all(np.isfinite(value).all() for value in after)
    assert any(not np.array_equal(old, new) for old, new in zip(before, after))
    assert cfg == original_cfg
