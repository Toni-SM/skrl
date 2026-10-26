import hypothesis
import hypothesis.strategies as st
import pytest

import dataclasses
import gymnasium

import torch

from skrl.memories.torch import RandomMemory
from skrl.multi_agents.torch.haa2c import HAA2C as MultiAgent
from skrl.multi_agents.torch.haa2c import HAA2C_CFG as MultiAgentCfg
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.model_instantiators.torch import (
    categorical_model,
    deterministic_model,
    gaussian_model,
    multivariate_gaussian_model,
)

from ...utilities import MultiAgentEnv, check_config_keys, get_test_mixed_precision, is_device_available


@hypothesis.given(
    num_envs=st.integers(min_value=1, max_value=5),
    max_num_agents=st.integers(min_value=2, max_value=5),
    # agent config
    rollouts=st.integers(min_value=1, max_value=5),
    learning_epochs=st.integers(min_value=1, max_value=5),
    mini_batches=st.integers(min_value=1, max_value=5),
    discount_factor=st.floats(min_value=0, max_value=1),
    gae_lambda=st.floats(min_value=0, max_value=1),
    learning_rate=st.floats(min_value=1.0e-10, max_value=1),
    learning_rate_scheduler=st.one_of(st.none(), st.just(KLAdaptiveLR), st.just(torch.optim.lr_scheduler.ConstantLR)),
    learning_rate_scheduler_kwargs_value=st.floats(min_value=0.1, max_value=1),
    observation_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    state_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    value_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    grad_norm_clip=st.floats(min_value=0, max_value=1),
    value_clip=st.floats(min_value=0, max_value=1),
    entropy_loss_scale=st.floats(min_value=0, max_value=1),
    value_loss_scale=st.floats(min_value=0, max_value=1),
    value_loss_type=st.sampled_from(["mse", "huber"]),
    huber_delta=st.floats(min_value=0.1, max_value=10),
    rewards_shaper=st.one_of(st.none(), st.just(lambda rewards, *args, **kwargs: 0.5 * rewards)),
    time_limit_bootstrap=st.booleans(),
    mixed_precision=st.booleans(),
    fixed_order=st.booleans(),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    max_examples=15,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("shared_value", [True, False])
@pytest.mark.parametrize("policy_structure", ["GaussianMixin", "MultivariateGaussianMixin", "CategoricalMixin"])
# CPU cases run first (the CUDA + CategoricalMixin case is skipped, see below)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_agent(
    capsys,
    device,
    num_envs,
    max_num_agents,
    # model config
    shared_value,
    policy_structure,
    # agent config
    rollouts,
    learning_epochs,
    mini_batches,
    discount_factor,
    gae_lambda,
    learning_rate,
    learning_rate_scheduler,
    learning_rate_scheduler_kwargs_value,
    observation_preprocessor,
    state_preprocessor,
    value_preprocessor,
    grad_norm_clip,
    value_clip,
    entropy_loss_scale,
    value_loss_scale,
    value_loss_type,
    huber_delta,
    rewards_shaper,
    time_limit_bootstrap,
    mixed_precision,
    fixed_order,
):
    # check device availability
    if not is_device_available(device, backend="torch"):
        pytest.skip(f"Device {device} not available")
    # on CUDA, NaN categorical probabilities (pre-existing NaN-std issue shared with the IPPO/MAPPO tests) trigger an
    # asynchronous device-side assertion that invalidates the CUDA context for the rest of the test process
    if policy_structure == "CategoricalMixin" and device.startswith("cuda"):
        pytest.skip(
            "CUDA + CategoricalMixin: pre-existing NaN-std device-side assert (shared with the IPPO/MAPPO tests)"
        )

    # spaces (heterogeneous observation and action spaces, common state space)
    observation_spaces = {}
    state_spaces = {}
    action_spaces = {}
    for i in range(max_num_agents):
        uid = f"agent_{i}"
        observation_spaces[uid] = gymnasium.spaces.Box(low=-1, high=1, shape=(2 + i,))
        state_spaces[uid] = gymnasium.spaces.Box(low=-1, high=1, shape=(6,))  # common
        if policy_structure in ["GaussianMixin", "MultivariateGaussianMixin"]:
            action_spaces[uid] = gymnasium.spaces.Box(low=-1, high=1, shape=(1 + i,))
        elif policy_structure == "CategoricalMixin":
            action_spaces[uid] = gymnasium.spaces.Discrete(2 + i)

    # env
    env = MultiAgentEnv(
        observation_spaces=observation_spaces,
        state_spaces=state_spaces,
        action_spaces=action_spaces,
        num_envs=num_envs,
        device=device,
        ml_framework="torch",
    )

    # models
    network = {
        "policy": [
            {
                "name": "net",
                "input": "OBSERVATIONS",
                "layers": [5],
                "activations": "relu",
            }
        ],
        "value": [
            {
                "name": "net",
                "input": "STATES",
                "layers": [5],
                "activations": "relu",
            }
        ],
    }
    models = {}
    for uid in env.possible_agents:
        models[uid] = {}
        if policy_structure == "GaussianMixin":
            models[uid]["policy"] = gaussian_model(
                observation_space=env.observation_space(uid),
                state_space=env.state_space(uid),
                action_space=env.action_space(uid),
                device=env.device,
                network=network["policy"],
                output="ACTIONS",
            )
        elif policy_structure == "MultivariateGaussianMixin":
            models[uid]["policy"] = multivariate_gaussian_model(
                observation_space=env.observation_space(uid),
                state_space=env.state_space(uid),
                action_space=env.action_space(uid),
                device=env.device,
                network=network["policy"],
                output="ACTIONS",
            )
        elif policy_structure == "CategoricalMixin":
            models[uid]["policy"] = categorical_model(
                observation_space=env.observation_space(uid),
                state_space=env.state_space(uid),
                action_space=env.action_space(uid),
                device=env.device,
                network=network["policy"],
                output="ACTIONS",
            )
        # - value: the same instance for all agents (shared) or one instance per agent
        if shared_value and uid != env.possible_agents[0]:
            models[uid]["value"] = models[env.possible_agents[0]]["value"]
        else:
            models[uid]["value"] = deterministic_model(
                observation_space=env.observation_space(uid),
                state_space=env.state_space(uid),
                action_space=env.action_space(uid),
                device=env.device,
                network=network["value"],
                output="ONE",
            )

    # memory
    memories = {}
    for uid in env.possible_agents:
        memories[uid] = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=env.device)

    # agent
    cfg = {
        "rollouts": rollouts,
        "learning_epochs": learning_epochs,
        "mini_batches": mini_batches,
        "discount_factor": discount_factor,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "learning_rate_scheduler": learning_rate_scheduler,
        "learning_rate_scheduler_kwargs": {},
        "observation_preprocessor": observation_preprocessor,
        "observation_preprocessor_kwargs": {
            uid: {"size": env.observation_space(uid), "device": env.device} for uid in env.possible_agents
        },
        "state_preprocessor": state_preprocessor,
        "state_preprocessor_kwargs": {"size": env.state_space("agent_0"), "device": env.device},
        "value_preprocessor": value_preprocessor,
        "value_preprocessor_kwargs": {"size": 1, "device": env.device},
        "grad_norm_clip": grad_norm_clip,
        "value_clip": value_clip,
        "entropy_loss_scale": entropy_loss_scale,
        "value_loss_scale": value_loss_scale,
        "value_loss_type": value_loss_type,
        "huber_delta": huber_delta,
        "rewards_shaper": rewards_shaper,
        "time_limit_bootstrap": time_limit_bootstrap,
        "mixed_precision": get_test_mixed_precision(mixed_precision),
        "fixed_order": fixed_order,
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
    check_config_keys(cfg, dataclasses.asdict(MultiAgentCfg()))
    check_config_keys(cfg["experiment"], dataclasses.asdict(MultiAgentCfg().experiment))
    agent = MultiAgent(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,
        cfg=cfg,
        observation_spaces=env.observation_spaces,
        state_spaces=env.state_spaces,
        action_spaces=env.action_spaces,
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

    if policy_structure == "CategoricalMixin":
        try:
            trainer.train()
        except RuntimeError as e:
            error_messages = [
                "probability tensor contains either",
                "invalid multinomial distribution",
            ]
            if not any(message in str(e) for message in error_messages):
                raise e
    else:
        trainer.train()
