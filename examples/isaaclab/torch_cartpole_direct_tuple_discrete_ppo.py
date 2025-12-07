import argparse
import os

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl import logger
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.spaces.torch import compute_space_size, unflatten_tensorized_space


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (logging/checkpointing disabled)")


# define shared model (stochastic and deterministic models) using mixins
class Shared(CategoricalMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        CategoricalMixin.__init__(self, unnormalized_log_prob=unnormalized_log_prob)
        DeterministicMixin.__init__(self)

        self.net_pos = nn.Sequential(
            nn.Linear(compute_space_size(self.observation_space[0]), 16),
            nn.ELU(),
            nn.Linear(16, 16),
            nn.ELU(),
        )
        self.net_vel = nn.Sequential(
            nn.Linear(compute_space_size(self.observation_space[1]), 16),
            nn.ELU(),
            nn.Linear(16, 16),
            nn.ELU(),
        )

        self.value_layer = nn.Linear(16, 1)
        self.policy_layer = nn.Linear(16, self.num_actions)

    def act(self, inputs, role):
        if role == "policy":
            return CategoricalMixin.act(self, inputs, role=role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role=role)

    def compute(self, inputs, role):
        observations = unflatten_tensorized_space(self.observation_space, inputs["observations"])
        if role == "policy":
            self._shared_output = self.net_pos(observations[0]) * self.net_vel(observations[1])
            return self.policy_layer(self._shared_output), {}
        elif role == "value":
            shared_output = (
                self.net_pos(observations[0]) * self.net_vel(observations[1])
                if self._shared_output is None
                else self._shared_output
            )
            self._shared_output = None
            return self.value_layer(shared_output), {}


# load the environment
task_name = "Isaac-Cartpole-Showcase-Tuple-Discrete-Direct-v0"
env = load_isaaclab_env(task_name=task_name, parser=parser)
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
models["policy"] = Shared(env.observation_space, env.state_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model


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
cfg.kl_threshold = 0
cfg.rewards_shaper = lambda rewards, *args, **kwargs: rewards * 0.1
cfg.time_limit_bootstrap = False
cfg.mixed_precision = False
# logging to TensorBoard and write checkpoints (in timesteps)
cfg.experiment.write_interval = "auto" if not args.eval else 0
cfg.experiment.checkpoint_interval = "auto" if not args.eval else 0
cfg.experiment.directory = f"runs/torch/{task_name}"

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
cfg_trainer = {"timesteps": 1600, "headless": args.headless}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: '{args.checkpoint}'")
        exit(1)
    agent.load(args.checkpoint)

trainer.train() if not args.eval else trainer.eval()
