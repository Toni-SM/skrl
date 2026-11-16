import pytest

import functools
import gymnasium

import numpy as np

from skrl.resources.schedulers.warp import KLAdaptiveLR
from skrl.utils.runner.warp import Runner

from ...utilities import SingleAgentEnv


class Schedule:
    def __call__(self, count):
        return 0.001


class SchedulerFactory:
    def __init__(self, adaptive):
        self.adaptive = adaptive

    def __call__(self, **kwargs):
        return KLAdaptiveLR(**kwargs) if self.adaptive else Schedule()


def scheduler_config(kind):
    kwargs = {"kl_threshold": 1.0, "min_lr": 0.0005, "max_lr": 0.0005}
    if kind == "direct":
        return KLAdaptiveLR, kwargs
    if kind == "partial":
        return functools.partial(KLAdaptiveLR, **kwargs), {}
    if kind == "callable":
        return SchedulerFactory(adaptive=True), kwargs
    if kind == "wrapper":
        return lambda **kw: KLAdaptiveLR(**kw), kwargs
    if kind == "ordinary_partial":
        return functools.partial(SchedulerFactory(adaptive=False)), {}
    return SchedulerFactory(adaptive=False), {}


@pytest.mark.parametrize("algorithm", ["PPO"])
@pytest.mark.parametrize("kind", ["direct", "partial", "callable", "wrapper", "ordinary_partial", "ordinary_callable"])
def test_callable_scheduler_factory_trains(algorithm, kind):
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(2,))
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,))
    env = SingleAgentEnv(
        observation_space=observation_space,
        state_space=observation_space,
        action_space=action_space,
        num_envs=2,
        device="cpu",
        ml_framework="warp",
        probability=0,
    )
    scheduler, scheduler_kwargs = scheduler_config(kind)
    network = [{"name": "net", "input": "OBSERVATIONS", "layers": [8], "activations": "tanh"}]
    cfg = {
        "seed": 17,
        "models": {
            "separate": True,
            "policy": {"class": "GaussianMixin", "clip_actions": True, "network": network, "output": "ACTIONS"},
            "value": {"class": "DeterministicMixin", "network": network, "output": "ONE"},
        },
        "memory": {"class": "RandomMemory", "memory_size": 4},
        "agent": {
            "class": algorithm,
            "rollouts": 4,
            "mini_batches": 1,
            "learning_rate": 0.001,
            "learning_rate_scheduler": scheduler,
            "learning_rate_scheduler_kwargs": scheduler_kwargs,
            "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        },
        "trainer": {
            "class": "SequentialTrainer",
            "timesteps": 8,
            "headless": True,
            "disable_progressbar": True,
            "close_environment_at_exit": False,
        },
    }
    if algorithm != "A2C":
        cfg["agent"]["learning_epochs"] = 1
    runner = Runner(env, cfg)
    adaptive = not kind.startswith("ordinary")
    expected_type = KLAdaptiveLR if adaptive else None
    assert runner.agent.scheduler_type is expected_type
    models = [runner.agent.policy]
    before = [[x.numpy().copy() for x in model.parameters()] for model in models]
    runner.run("train")
    after = [[x.numpy().copy() for x in model.parameters()] for model in models]
    for old, new in zip(before, after):
        assert all(np.isfinite(value).all() for value in new)
        assert any(not np.array_equal(x, y) for x, y in zip(old, new))
    if adaptive:
        assert np.isclose(runner.agent.learning_rate, 0.0005)
