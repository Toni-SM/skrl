import argparse
import os
import gym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl import logger
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.resources.noises.torch import GaussianNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--headless", action="store_true", help="Run in headless mode (no rendering)")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (logging/checkpointing disabled)")
args, _ = parser.parse_known_args()


# seed for reproducibility
set_seed(args.seed)  # e.g. `set_seed(42)` for fixed seed


# define models (deterministic models) using mixin
class Actor(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, clip_actions=False):
        Model.__init__(
            self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, self.num_actions),
            nn.Tanh(),
        )

    def compute(self, inputs, role):
        x = self.net(inputs["observations"])
        # Pendulum-v1 action_space is -2 to 2
        return 2.0 * x, {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device
        )
        DeterministicMixin.__init__(self)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def compute(self, inputs, role):
        x = self.net(torch.cat([inputs["observations"], inputs["taken_actions"]], dim=1))
        return x, {}


# load the environment (note: the environment version may change depending on the gym version)
task_name = "Pendulum"
render_mode = "human" if not args.headless else None
env_id = [spec for spec in gym.envs.registry if spec.startswith(f"{task_name}-v")][-1]  # get latest environment version
if args.num_envs <= 1:
    env = gym.make(env_id, render_mode=render_mode)
else:
    env = gym.vector.make(env_id, num_envs=args.num_envs, render_mode=render_mode, asynchronous=False)
# wrap the environment
env = wrap_env(env)

device = env.device


# instantiate a memory as experience replay
memory = RandomMemory(memory_size=15000, num_envs=env.num_envs, device=device, replacement=False)


# instantiate the agent's models (function approximators).
# TD3 requires 6 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/td3.html#models
models = {}
models["policy"] = Actor(env.observation_space, env.state_space, env.action_space, device)
models["target_policy"] = Actor(env.observation_space, env.state_space, env.action_space, device)
models["critic_1"] = Critic(env.observation_space, env.state_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.state_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.state_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.state_space, env.action_space, device)

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/td3.html#configuration-and-hyperparameters
cfg = TD3_DEFAULT_CONFIG.copy()
cfg["exploration"]["noise"] = GaussianNoise
cfg["exploration"]["noise_kwargs"] = {"mean": 0.0, "std": 0.1, "device": device}
cfg["smooth_regularization_noise"] = GaussianNoise
cfg["smooth_regularization_noise_kwargs"] = {"mean": 0.0, "std": 0.2, "device": device}
cfg["smooth_regularization_clip"] = 0.5
cfg["discount_factor"] = 0.98
cfg["batch_size"] = 100
cfg["random_timesteps"] = 100
cfg["learning_starts"] = 100
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = "auto" if not args.eval else 0
cfg["experiment"]["checkpoint_interval"] = "auto" if not args.eval else 0
cfg["experiment"]["directory"] = f"runs/torch/{task_name}"

agent = TD3(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 15000, "headless": args.headless}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: '{args.checkpoint}'")
        exit(1)
    agent.load(args.checkpoint)

trainer.train() if not args.eval else trainer.eval()
