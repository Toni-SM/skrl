import hypothesis
import hypothesis.strategies as st
import pytest

import gymnasium

import torch

from skrl.agents.torch.trpo import TRPO as Agent
from skrl.agents.torch.trpo import TRPO_DEFAULT_CONFIG as DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model, multivariate_gaussian_model
from skrl.utils.spaces.torch import sample_space

from ...utils import BaseEnv, get_test_mixed_precision, is_device_available


class Env(BaseEnv):
    def _sample_observation(self):
        return sample_space(self.observation_space, batch_size=self.num_envs, backend="numpy")


def _check_agent_config(config, default_config):
    for k in config.keys():
        assert k in default_config
        if k == "experiment":
            _check_agent_config(config["experiment"], default_config["experiment"])
    for k in default_config.keys():
        assert k in config
        if k == "experiment":
            _check_agent_config(config["experiment"], default_config["experiment"])


@hypothesis.given(
    num_envs=st.integers(min_value=1, max_value=5),
    rollouts=st.integers(min_value=1, max_value=5),
    learning_epochs=st.integers(min_value=1, max_value=5),
    mini_batches=st.integers(min_value=1, max_value=5),
    discount_factor=st.floats(min_value=0, max_value=1),
    lambda_=st.floats(min_value=0, max_value=1),
    value_learning_rate=st.floats(min_value=1.0e-10, max_value=1),
    learning_rate_scheduler=st.one_of(st.none(), st.just(KLAdaptiveLR), st.just(torch.optim.lr_scheduler.ConstantLR)),
    learning_rate_scheduler_kwargs_value=st.floats(min_value=0.1, max_value=1),
    state_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    value_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    random_timesteps=st.just(0),
    learning_starts=st.just(0),
    grad_norm_clip=st.floats(min_value=0, max_value=1),
    value_loss_scale=st.floats(min_value=0, max_value=1),
    damping=st.floats(min_value=0, max_value=1),
    max_kl_divergence=st.floats(min_value=0, max_value=1),
    conjugate_gradient_steps=st.integers(min_value=1, max_value=5),
    max_backtrack_steps=st.integers(min_value=1, max_value=5),
    accept_ratio=st.floats(min_value=0, max_value=1),
    step_fraction=st.floats(min_value=0, max_value=1),
    rewards_shaper=st.one_of(st.none(), st.just(lambda rewards, *args, **kwargs: 0.5 * rewards)),
    time_limit_bootstrap=st.booleans(),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("policy_structure", ["GaussianMixin", "MultivariateGaussianMixin"])
def test_agent(
    capsys,
    device,
    num_envs,
    # model config
    policy_structure,
    # agent config
    rollouts,
    learning_epochs,
    mini_batches,
    discount_factor,
    lambda_,
    value_learning_rate,
    learning_rate_scheduler,
    learning_rate_scheduler_kwargs_value,
    state_preprocessor,
    value_preprocessor,
    random_timesteps,
    learning_starts,
    grad_norm_clip,
    value_loss_scale,
    damping,
    max_kl_divergence,
    conjugate_gradient_steps,
    max_backtrack_steps,
    accept_ratio,
    step_fraction,
    rewards_shaper,
    time_limit_bootstrap,
):
    # check device availability
    if not is_device_available(device, backend="torch"):
        pytest.skip(f"Device {device} not available")

    # spaces
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(5,))
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(3,))

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
    if policy_structure == "GaussianMixin":
        models["policy"] = gaussian_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            network=network,
            output="ACTIONS",
        )
    elif policy_structure == "MultivariateGaussianMixin":
        models["policy"] = multivariate_gaussian_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            network=network,
            output="ACTIONS",
        )
    models["value"] = deterministic_model(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        network=network,
        output="ONE",
    )

    # memory
    memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=env.device)

    # agent
    cfg = {
        "rollouts": rollouts,
        "learning_epochs": learning_epochs,
        "mini_batches": mini_batches,
        "discount_factor": discount_factor,
        "lambda": lambda_,
        "value_learning_rate": value_learning_rate,
        "learning_rate_scheduler": learning_rate_scheduler,
        "learning_rate_scheduler_kwargs": {},
        "state_preprocessor": state_preprocessor,
        "state_preprocessor_kwargs": {"size": env.observation_space, "device": env.device},
        "value_preprocessor": value_preprocessor,
        "value_preprocessor_kwargs": {"size": 1, "device": env.device},
        "random_timesteps": random_timesteps,
        "learning_starts": learning_starts,
        "grad_norm_clip": grad_norm_clip,
        "value_loss_scale": value_loss_scale,
        "damping": damping,
        "max_kl_divergence": max_kl_divergence,
        "conjugate_gradient_steps": conjugate_gradient_steps,
        "max_backtrack_steps": max_backtrack_steps,
        "accept_ratio": accept_ratio,
        "step_fraction": step_fraction,
        "rewards_shaper": rewards_shaper,
        "time_limit_bootstrap": time_limit_bootstrap,
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

    trainer.train()
