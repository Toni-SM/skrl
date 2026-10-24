import hypothesis
import hypothesis.strategies as st
import pytest

import dataclasses
import gymnasium

import torch

from skrl.memories.torch import RandomMemory
from skrl.multi_agents.torch.matd3 import MATD3 as MultiAgent
from skrl.multi_agents.torch.matd3 import MATD3_CFG as MultiAgentCfg
from skrl.resources.noises.torch import GaussianNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.model_instantiators.torch import deterministic_model
from skrl.utils.spaces.torch import compute_joint_space

from ...utilities import MultiAgentEnv, check_config_keys, get_test_mixed_precision, is_device_available


@hypothesis.given(
    num_envs=st.integers(min_value=2, max_value=5),
    max_num_agents=st.integers(min_value=2, max_value=4),
    # agent config
    gradient_steps=st.integers(min_value=1, max_value=2),
    batch_size=st.integers(min_value=2, max_value=5),
    discount_factor=st.floats(min_value=0, max_value=1),
    polyak=st.floats(min_value=0, max_value=1),
    learning_rate=st.one_of(
        st.floats(min_value=1.0e-10, max_value=1),
        st.tuples(st.floats(min_value=1.0e-10, max_value=1), st.floats(min_value=1.0e-10, max_value=1)),
    ),
    learning_rate_scheduler=st.one_of(st.none(), st.just(KLAdaptiveLR), st.just(torch.optim.lr_scheduler.ConstantLR)),
    learning_rate_scheduler_kwargs_value=st.floats(min_value=0.1, max_value=1),
    observation_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    state_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    random_timesteps=st.integers(min_value=0, max_value=5),
    learning_starts=st.integers(min_value=0, max_value=5),
    grad_norm_clip=st.floats(min_value=0, max_value=1),
    exploration_noise=st.one_of(st.none(), st.just(GaussianNoise)),
    exploration_scheduler=st.one_of(st.none(), st.just(lambda timestep, timesteps: 1.0 - timestep / timesteps)),
    policy_delay=st.integers(min_value=1, max_value=3),
    smooth_regularization_noise=st.one_of(st.none(), st.just(GaussianNoise)),
    smooth_regularization_clip=st.floats(min_value=0, max_value=1),
    rewards_shaper=st.one_of(st.none(), st.just(lambda rewards, *args, **kwargs: 0.5 * rewards)),
    mixed_precision=st.booleans(),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    max_examples=15,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("shared_critic", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_agent(
    capsys,
    device,
    num_envs,
    max_num_agents,
    # model config
    shared_critic,
    # agent config
    gradient_steps,
    batch_size,
    discount_factor,
    polyak,
    learning_rate,
    learning_rate_scheduler,
    learning_rate_scheduler_kwargs_value,
    observation_preprocessor,
    state_preprocessor,
    random_timesteps,
    learning_starts,
    grad_norm_clip,
    exploration_noise,
    exploration_scheduler,
    policy_delay,
    smooth_regularization_noise,
    smooth_regularization_clip,
    rewards_shaper,
    mixed_precision,
):
    # check device availability
    if not is_device_available(device, backend="torch"):
        pytest.skip(f"Device {device} not available")

    # spaces (heterogeneous observation and action spaces, common state space)
    observation_spaces = {}
    state_spaces = {}
    action_spaces = {}
    for i in range(max_num_agents):
        uid = f"agent_{i}"
        observation_spaces[uid] = gymnasium.spaces.Box(low=-1, high=1, shape=(2 + i,))
        state_spaces[uid] = gymnasium.spaces.Box(low=-1, high=1, shape=(6,))  # common
        action_spaces[uid] = gymnasium.spaces.Box(low=-1, high=1, shape=(1 + i,))

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
        "critic": [
            {
                "name": "net",
                "input": "concatenate([STATES, ACTIONS])",
                "layers": [5],
                "activations": "relu",
            }
        ],
    }
    joint_action_space = compute_joint_space([env.action_space(uid) for uid in env.possible_agents], mode="concatenate")
    models = {}
    for uid in env.possible_agents:
        models[uid] = {}
        for role in ["policy", "target_policy"]:
            models[uid][role] = deterministic_model(
                observation_space=env.observation_space(uid),
                state_space=env.state_space(uid),
                action_space=env.action_space(uid),
                device=env.device,
                network=network["policy"],
                output="ACTIONS",
            )
        # - critics: built with the joint action space, the same instance for all agents (shared) or one per agent
        for role in ["critic_1", "critic_2", "target_critic_1", "target_critic_2"]:
            if shared_critic and uid != env.possible_agents[0]:
                models[uid][role] = models[env.possible_agents[0]][role]
            else:
                models[uid][role] = deterministic_model(
                    observation_space=env.observation_space(uid),
                    state_space=env.state_space(uid),
                    action_space=joint_action_space,
                    device=env.device,
                    network=network["critic"],
                    output="ONE",
                )

    # memory
    memories = {}
    for uid in env.possible_agents:
        memories[uid] = RandomMemory(memory_size=50, num_envs=env.num_envs, device=env.device)

    # agent
    cfg = {
        "gradient_steps": gradient_steps,
        "batch_size": batch_size,
        "discount_factor": discount_factor,
        "polyak": polyak,
        "learning_rate": learning_rate,
        "learning_rate_scheduler": learning_rate_scheduler,
        "learning_rate_scheduler_kwargs": {},
        "observation_preprocessor": observation_preprocessor,
        "observation_preprocessor_kwargs": {
            uid: {"size": env.observation_space(uid), "device": env.device} for uid in env.possible_agents
        },
        "state_preprocessor": state_preprocessor,
        "state_preprocessor_kwargs": {"size": env.state_space("agent_0"), "device": env.device},
        "random_timesteps": random_timesteps,
        "learning_starts": learning_starts,
        "grad_norm_clip": grad_norm_clip,
        "exploration_noise": exploration_noise,
        "exploration_noise_kwargs": {},
        "exploration_scheduler": exploration_scheduler,
        "policy_delay": policy_delay,
        "smooth_regularization_noise": smooth_regularization_noise,
        "smooth_regularization_noise_kwargs": {},
        "smooth_regularization_clip": smooth_regularization_clip,
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
    # noise
    # - exploration
    if exploration_noise is GaussianNoise:
        cfg["exploration_noise_kwargs"] = {"mean": 0, "std": 0.1, "device": env.device}
    # - smooth regularization
    if smooth_regularization_noise is GaussianNoise:
        cfg["smooth_regularization_noise_kwargs"] = {"mean": 0, "std": 0.2, "device": env.device}
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
        "timesteps": 20,
        "headless": True,
        "disable_progressbar": True,
        "close_environment_at_exit": False,
    }
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    trainer.train()
