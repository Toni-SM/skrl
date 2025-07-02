import hypothesis
import hypothesis.strategies as st
import pytest

import gymnasium

import torch

from skrl.agents.torch.amp import AMP as Agent
from skrl.agents.torch.amp import AMP_DEFAULT_CONFIG as DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model
from skrl.utils.spaces.torch import sample_space

from ...utilities import SingleAgentEnv, check_config_keys, get_test_mixed_precision, is_device_available


class CustomSingleAgentEnv(SingleAgentEnv):
    def __init__(
        self, *, observation_space, state_space, action_space, num_envs, device, ml_framework, amp_observation_space
    ):
        super().__init__(
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            num_envs=num_envs,
            device=device,
            ml_framework=ml_framework,
        )
        self.amp_observation_space = amp_observation_space

    def step(self, actions):
        observations, rewards, terminated, truncated, info = super().step(actions)
        info["terminate"] = torch.tensor(terminated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)
        info["amp_obs"] = sample_space(
            self.amp_observation_space, batch_size=self.num_envs, backend="native", device=self.device
        )
        return observations, rewards, terminated, truncated, info

    def fetch_amp_obs_demo(self, num_samples):
        return sample_space(self.amp_observation_space, batch_size=num_samples, backend="native", device=self.device)

    def reset_done(self):
        return (
            {
                "obs": sample_space(
                    self.observation_space, batch_size=self.num_envs, backend="native", device=self.device
                )
            },
        )


@hypothesis.given(
    num_envs=st.integers(min_value=1, max_value=5),
    # agent config
    rollouts=st.integers(min_value=1, max_value=5),
    learning_epochs=st.integers(min_value=1, max_value=5),
    mini_batches=st.integers(min_value=1, max_value=5),
    discount_factor=st.floats(min_value=0, max_value=1),
    lambda_=st.floats(min_value=0, max_value=1),
    learning_rate=st.floats(min_value=1.0e-10, max_value=1),
    learning_rate_scheduler=st.one_of(st.none(), st.just(KLAdaptiveLR), st.just(torch.optim.lr_scheduler.ConstantLR)),
    learning_rate_scheduler_kwargs_value=st.floats(min_value=0.1, max_value=1),
    observation_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    state_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    value_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    amp_observation_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    random_timesteps=st.just(0),
    learning_starts=st.just(0),
    grad_norm_clip=st.floats(min_value=0, max_value=1),
    ratio_clip=st.floats(min_value=0, max_value=1),
    value_clip=st.floats(min_value=0, max_value=1),
    clip_predicted_values=st.booleans(),
    entropy_loss_scale=st.floats(min_value=0, max_value=1),
    value_loss_scale=st.floats(min_value=0, max_value=1),
    discriminator_loss_scale=st.floats(min_value=0, max_value=1),
    amp_batch_size=st.integers(min_value=1, max_value=5),
    task_reward_weight=st.floats(min_value=0, max_value=1),
    style_reward_weight=st.floats(min_value=0, max_value=1),
    discriminator_batch_size=st.integers(min_value=0, max_value=5),
    discriminator_reward_scale=st.floats(min_value=0, max_value=1),
    discriminator_logit_regularization_scale=st.floats(min_value=0, max_value=1),
    discriminator_gradient_penalty_scale=st.floats(min_value=0, max_value=1),
    discriminator_weight_decay_scale=st.floats(min_value=0, max_value=1),
    rewards_shaper=st.one_of(st.none(), st.just(lambda rewards, *args, **kwargs: 0.5 * rewards)),
    time_limit_bootstrap=st.booleans(),
    mixed_precision=st.booleans(),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("separate", [True])
@pytest.mark.parametrize("asymmetric", [True, False])
@pytest.mark.parametrize("policy_structure", ["GaussianMixin"])
def test_agent(
    capsys,
    device,
    num_envs,
    asymmetric,
    # model config
    separate,
    policy_structure,
    # agent config
    rollouts,
    learning_epochs,
    mini_batches,
    discount_factor,
    lambda_,
    learning_rate,
    learning_rate_scheduler,
    learning_rate_scheduler_kwargs_value,
    observation_preprocessor,
    state_preprocessor,
    value_preprocessor,
    amp_observation_preprocessor,
    random_timesteps,
    learning_starts,
    grad_norm_clip,
    ratio_clip,
    value_clip,
    clip_predicted_values,
    entropy_loss_scale,
    value_loss_scale,
    discriminator_loss_scale,
    amp_batch_size,
    task_reward_weight,
    style_reward_weight,
    discriminator_batch_size,
    discriminator_reward_scale,
    discriminator_logit_regularization_scale,
    discriminator_gradient_penalty_scale,
    discriminator_weight_decay_scale,
    rewards_shaper,
    time_limit_bootstrap,
    mixed_precision,
):
    # check device availability
    if not is_device_available(device, backend="torch"):
        pytest.skip(f"Device {device} not available")

    # spaces
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,))
    state_space = gymnasium.spaces.Box(low=-1, high=1, shape=(5,)) if asymmetric else None
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(3,))
    amp_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(10,))

    # env
    env = CustomSingleAgentEnv(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        num_envs=num_envs,
        device=device,
        ml_framework="torch",
        amp_observation_space=amp_observation_space,
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
                "input": "STATES" if asymmetric else "OBSERVATIONS",
                "layers": [5],
                "activations": "relu",
            }
        ],
        "discriminator": [
            {
                "name": "net",
                "input": "OBSERVATIONS",
                "layers": [5],
                "activations": "relu",
            }
        ],
    }
    models = {}
    if separate:
        models["policy"] = gaussian_model(
            observation_space=env.observation_space,
            state_space=env.state_space,
            action_space=env.action_space,
            device=env.device,
            network=network["policy"],
            output="ACTIONS",
        )
        models["value"] = deterministic_model(
            observation_space=env.observation_space,
            state_space=env.state_space,
            action_space=env.action_space,
            device=env.device,
            network=network["value"],
            output="ONE",
        )
        models["discriminator"] = deterministic_model(
            observation_space=env.amp_observation_space,
            state_space=env.state_space,
            action_space=env.action_space,
            device=env.device,
            network=network["discriminator"],
            output="ONE",
        )
    else:
        raise ValueError

    # memory
    memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=env.device)

    # agent
    cfg = {
        "rollouts": rollouts,
        "learning_epochs": learning_epochs,
        "mini_batches": mini_batches,
        "discount_factor": discount_factor,
        "lambda": lambda_,
        "learning_rate": learning_rate,
        "learning_rate_scheduler": learning_rate_scheduler,
        "learning_rate_scheduler_kwargs": {},
        "observation_preprocessor": observation_preprocessor,
        "observation_preprocessor_kwargs": {"size": env.observation_space, "device": env.device},
        "state_preprocessor": state_preprocessor,
        "state_preprocessor_kwargs": {"size": env.state_space, "device": env.device},
        "value_preprocessor": value_preprocessor,
        "value_preprocessor_kwargs": {"size": 1, "device": env.device},
        "amp_observation_preprocessor": amp_observation_preprocessor,
        "amp_observation_preprocessor_kwargs": {"size": env.amp_observation_space, "device": env.device},
        "random_timesteps": random_timesteps,
        "learning_starts": learning_starts,
        "grad_norm_clip": grad_norm_clip,
        "ratio_clip": ratio_clip,
        "value_clip": value_clip,
        "clip_predicted_values": clip_predicted_values,
        "entropy_loss_scale": entropy_loss_scale,
        "value_loss_scale": value_loss_scale,
        "discriminator_loss_scale": discriminator_loss_scale,
        "amp_batch_size": amp_batch_size,
        "task_reward_weight": task_reward_weight,
        "style_reward_weight": style_reward_weight,
        "discriminator_batch_size": discriminator_batch_size,
        "discriminator_reward_scale": discriminator_reward_scale,
        "discriminator_logit_regularization_scale": discriminator_logit_regularization_scale,
        "discriminator_gradient_penalty_scale": discriminator_gradient_penalty_scale,
        "discriminator_weight_decay_scale": discriminator_weight_decay_scale,
        "rewards_shaper": rewards_shaper,
        "time_limit_bootstrap": time_limit_bootstrap,
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
    check_config_keys(cfg, DEFAULT_CONFIG)
    check_config_keys(cfg["experiment"], DEFAULT_CONFIG["experiment"])
    agent = Agent(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
        amp_observation_space=env.amp_observation_space,
        motion_dataset=RandomMemory(memory_size=50, device=device),
        reply_buffer=RandomMemory(memory_size=100, device=device),
        collect_reference_motions=lambda num_samples: env.fetch_amp_obs_demo(num_samples),
        collect_observation=lambda: env.reset_done()[0]["obs"],
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
