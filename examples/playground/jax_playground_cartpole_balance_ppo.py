import argparse
import os

import flax.linen as nn
import jax.numpy as jnp

# import the skrl components to build the RL system
from skrl import logger
from skrl.agents.jax.ppo import PPO, PPO_CFG
from skrl.envs.loaders.jax import load_playground_env
from skrl.envs.wrappers.jax import wrap_env
from skrl.memories.jax import RandomMemory
from skrl.models.jax import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.jax import RunningStandardScaler
from skrl.resources.schedulers.jax import KLAdaptiveLR
from skrl.trainers.jax import SequentialTrainer
from skrl.utils import set_seed


# parse arguments (some arguments are already defined by the loader)
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode (no rendering)")
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (logging/checkpointing disabled)")
args, _ = parser.parse_known_args()


# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
        **kwargs,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            **kwargs,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )

    @nn.compact  # marks the given module method allowing inlined submodules
    def __call__(self, inputs, role):
        x = nn.elu(nn.Dense(32)(inputs["observations"]))
        x = nn.elu(nn.Dense(32)(x))
        x = nn.tanh(nn.Dense(self.num_actions)(x))
        log_std = self.param("log_std", lambda _: jnp.zeros(self.num_actions))
        return x, {"log_std": log_std}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, **kwargs):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            **kwargs,
        )
        DeterministicMixin.__init__(self)

    @nn.compact  # marks the given module method allowing inlined submodules
    def __call__(self, inputs, role):
        x = nn.elu(nn.Dense(32)(inputs["observations"]))
        x = nn.elu(nn.Dense(32)(x))
        x = nn.Dense(1)(x)
        return x, {}


# load the environment
task_name = "CartpoleBalance"
env = load_playground_env(task_name=task_name, num_envs=1024, episode_length=300, parser=parser)
# wrap the environment
env = wrap_env(env)

device = env.device


# defer parsing of arguments to include loader arguments (run with --help to see all the arguments)
args, _ = parser.parse_known_args()


# seed for reproducibility
set_seed(args.seed)  # e.g. `set_seed(42)` for fixed seed


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=32, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Policy(env.observation_space, env.state_space, env.action_space, device, clip_actions=True)
models["value"] = Value(env.observation_space, env.state_space, env.action_space, device)

# instantiate models' state dict
for role, model in models.items():
    model.init_state_dict(role=role)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_CFG()
cfg.rollouts = 32  # memory_size
cfg.learning_epochs = 8
cfg.mini_batches = 8
cfg.discount_factor = 0.99
cfg.lambda_ = 0.95
cfg.learning_rate = 5e-4
cfg.learning_rate_scheduler = KLAdaptiveLR
cfg.learning_rate_scheduler_kwargs = {"kl_threshold": 0.008}
cfg.observation_preprocessor = RunningStandardScaler
cfg.observation_preprocessor_kwargs = {"size": env.observation_space, "device": device}
cfg.value_preprocessor = RunningStandardScaler
cfg.value_preprocessor_kwargs = {"size": 1, "device": device}
cfg.grad_norm_clip = 1.0
cfg.ratio_clip = 0.2
cfg.value_clip = 0.2
cfg.entropy_loss_scale = 0.0
cfg.value_loss_scale = 2.0
# logging to TensorBoard and write checkpoints (in timesteps)
cfg.experiment.write_interval = "auto" if not args.eval else 0
cfg.experiment.checkpoint_interval = "auto" if not args.eval else 0
cfg.experiment.directory = f"runs/jax/{task_name}"

agent = PPO(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 16000, "headless": args.headless}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: '{args.checkpoint}'")
        exit(1)
    agent.load(args.checkpoint)

trainer.train() if not args.eval else trainer.eval()
