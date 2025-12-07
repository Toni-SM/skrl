import argparse
import os

import flax.linen as nn
import jax.numpy as jnp

# import the skrl components to build the RL system
from skrl import config, logger
from skrl.agents.jax.ddpg import DDPG, DDPG_CFG
from skrl.envs.loaders.jax import load_isaaclab_env
from skrl.envs.wrappers.jax import wrap_env
from skrl.memories.jax import RandomMemory
from skrl.models.jax import DeterministicMixin, Model
from skrl.resources.noises.jax import OrnsteinUhlenbeckNoise
from skrl.resources.preprocessors.jax import RunningStandardScaler
from skrl.trainers.jax import SequentialTrainer
from skrl.utils import set_seed


config.jax.backend = "jax"  # or "numpy"


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (logging/checkpointing disabled)")


# define models (deterministic models) using mixins
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, clip_actions=False, **kwargs):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            **kwargs,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

    @nn.compact  # marks the given module method allowing inlined submodules
    def __call__(self, inputs, role):
        x = nn.relu(nn.Dense(512)(inputs["observations"]))
        x = nn.relu(nn.Dense(256)(x))
        x = nn.Dense(self.num_actions)(x)
        return nn.tanh(x), {}


class Critic(DeterministicMixin, Model):
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
        x = jnp.concatenate([inputs["observations"], inputs["taken_actions"]], axis=-1)
        x = nn.relu(nn.Dense(512)(x))
        x = nn.relu(nn.Dense(256)(x))
        x = nn.Dense(1)(x)
        return x, {}


# load the environment
task_name = "Isaac-Ant-Direct-v0"
env = load_isaaclab_env(task_name=task_name, parser=parser, num_envs=64)
# wrap the environment
env = wrap_env(env)

device = env.device


# defer parsing of arguments to include loader arguments (run with --help to see all the arguments)
args, _ = parser.parse_known_args()


# seed for reproducibility
set_seed(args.seed)  # e.g. `set_seed(42)` for fixed seed


# instantiate a replay memory
memory = RandomMemory(memory_size=16000, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#models
models = {}
models["policy"] = DeterministicActor(env.observation_space, env.state_space, env.action_space, device)
models["target_policy"] = DeterministicActor(env.observation_space, env.state_space, env.action_space, device)
models["critic"] = Critic(env.observation_space, env.state_space, env.action_space, device)
models["target_critic"] = Critic(env.observation_space, env.state_space, env.action_space, device)

# instantiate models' state dict
for role, model in models.items():
    model.init_state_dict(role=role)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#configuration-and-hyperparameters
cfg = DDPG_CFG()
cfg.exploration_noise = OrnsteinUhlenbeckNoise
cfg.exploration_noise_kwargs = {"theta": 0.15, "sigma": 0.1, "base_scale": 0.5, "device": device, "std": 2.0}
cfg.exploration_scheduler = lambda timestep, timesteps: max(1 - timestep / timesteps, 1e-2)
cfg.gradient_steps = 1
cfg.batch_size = 4096
cfg.discount_factor = 0.99
cfg.polyak = 0.005
cfg.learning_rate = 5e-4
cfg.random_timesteps = 50
cfg.learning_starts = 50
cfg.state_preprocessor = RunningStandardScaler
cfg.state_preprocessor_kwargs = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg.experiment.write_interval = "auto" if not args.eval else 0
cfg.experiment.checkpoint_interval = "auto" if not args.eval else 0
cfg.experiment.directory = f"runs/jax/{task_name}"

agent = DDPG(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 160000, "headless": args.headless}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: '{args.checkpoint}'")
        exit(1)
    agent.load(args.checkpoint)

trainer.train() if not args.eval else trainer.eval()
