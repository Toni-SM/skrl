import hypothesis
import hypothesis.strategies as st
import pytest

import gymnasium

import torch

from skrl.agents.torch.cem import CEM as Agent
from skrl.agents.torch.cem import CEM_DEFAULT_CONFIG as DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.model_instantiators.torch import categorical_model
from skrl.utils.spaces.torch import sample_space

from ...utilities import BaseEnv, get_test_mixed_precision, is_device_available


class Env(BaseEnv):
    def _sample_observation(self):
        return sample_space(self.observation_space, batch_size=self.num_envs, backend="numpy")


def _check_agent_config(config, default_config):
    for k in config.keys():
        assert k in default_config
    for k in default_config.keys():
        assert k in config


@hypothesis.given(
    num_envs=st.integers(min_value=1, max_value=5),
    rollouts=st.integers(min_value=1, max_value=5),
    percentile=st.floats(min_value=0, max_value=1),
    discount_factor=st.floats(min_value=0, max_value=1),
    learning_rate=st.floats(min_value=1.0e-10, max_value=1),
    learning_rate_scheduler=st.one_of(st.none(), st.just(KLAdaptiveLR), st.just(torch.optim.lr_scheduler.ConstantLR)),
    learning_rate_scheduler_kwargs_value=st.floats(min_value=0.1, max_value=1),
    state_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    random_timesteps=st.just(0),
    learning_starts=st.just(0),
    rewards_shaper=st.one_of(st.none(), st.just(lambda rewards, *args, **kwargs: 0.5 * rewards)),
    mixed_precision=st.booleans(),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_agent(
    capsys,
    device,
    num_envs,
    # agent config
    rollouts,
    percentile,
    discount_factor,
    learning_rate,
    learning_rate_scheduler,
    learning_rate_scheduler_kwargs_value,
    state_preprocessor,
    random_timesteps,
    learning_starts,
    rewards_shaper,
    mixed_precision,
):
    # check device availability
    if not is_device_available(device, backend="torch"):
        pytest.skip(f"Device {device} not available")

    # spaces
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(5,))
    action_space = gymnasium.spaces.Discrete(3)

    # env
    env = wrap_env(Env(observation_space, action_space, num_envs, device), wrapper="gymnasium")

    # models
    network = [
        {
            "name": "net",
            "input": "STATES",
            "layers": [64, 64],
            "activations": "elu",
        }
    ]
    models = {}
    models["policy"] = categorical_model(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        network=network,
        output="ACTIONS",
    )

    # memory
    memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=env.device)

    # agent
    cfg = {
        "rollouts": rollouts,
        "percentile": percentile,
        "discount_factor": discount_factor,
        "learning_rate": learning_rate,
        "learning_rate_scheduler": learning_rate_scheduler,
        "learning_rate_scheduler_kwargs": {},
        "state_preprocessor": state_preprocessor,
        "state_preprocessor_kwargs": {"size": env.observation_space, "device": env.device},
        "random_timesteps": random_timesteps,
        "learning_starts": learning_starts,
        "rewards_shaper": rewards_shaper,
        "mixed_precision": get_test_mixed_precision(mixed_precision),
        "experiment": {
            "directory": "",
            "experiment_name": "",
            "write_interval": 0,
            "checkpoint_interval": 0,
            "store_separately": False,
            "wandb": False,
            "wandb_kwargs": {},
        },
    }
    cfg["learning_rate_scheduler_kwargs"][
        "kl_threshold" if learning_rate_scheduler is KLAdaptiveLR else "factor"
    ] = learning_rate_scheduler_kwargs_value
    _check_agent_config(cfg, DEFAULT_CONFIG)
    _check_agent_config(cfg["experiment"], DEFAULT_CONFIG["experiment"])
    agent = Agent(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    # trainer
    cfg_trainer = {
        "timesteps": int(5 * rollouts),
        "headless": True,
        "disable_progressbar": True,
        "close_environment_at_exit": False,
    }
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    try:
        trainer.train()
    except RuntimeError as e:
        error_messages = [
            "probability tensor contains either",
            "invalid multinomial distribution",
        ]
        if not any(message in str(e) for message in error_messages):
            raise e
