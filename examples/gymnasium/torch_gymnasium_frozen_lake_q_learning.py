import argparse
import os
import gymnasium as gym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl import logger
from skrl.agents.torch.q_learning import Q_LEARNING, Q_LEARNING_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.models.torch import Model, TabularMixin
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


# define model (tabular model) using mixin
class EpilonGreedyPolicy(TabularMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, epsilon=0.1):
        Model.__init__(
            self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device
        )
        TabularMixin.__init__(self)

        self.epsilon = epsilon
        self.q_table = nn.Parameter(
            torch.ones((self.num_observations, self.num_actions), dtype=torch.float32, device=self.device),
            requires_grad=False,
        )

    def compute(self, inputs, role):
        actions = torch.argmax(self.q_table[inputs["observations"]], dim=-1, keepdim=False)

        # choose random actions for exploration according to epsilon
        indexes = (torch.rand(inputs["observations"].shape[0], device=self.device) < self.epsilon).nonzero().flatten()
        if indexes.numel():
            actions[indexes] = torch.randint(self.num_actions, (indexes.numel(), 1), device=self.device)
        return actions, {}


# load the environment (note: the environment version may change depending on the gymnasium version)
task_name = "FrozenLake"
render_mode = "human" if not args.headless else None
env_id = [spec for spec in gym.envs.registry if spec.startswith(f"{task_name}-v")][-1]  # get latest environment version
if args.num_envs <= 1:
    env = gym.make(env_id, render_mode=render_mode)
else:
    env = gym.make_vec(env_id, num_envs=args.num_envs, render_mode=render_mode, vectorization_mode="sync")
# wrap the environment
env = wrap_env(env)

device = env.device


# instantiate the agent's model (table)
# Q-learning requires 1 model, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/q_learning.html#models
models = {}
models["policy"] = EpilonGreedyPolicy(env.observation_space, env.state_space, env.action_space, device, epsilon=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/q_learning.html#configuration-and-hyperparameters
cfg = Q_LEARNING_DEFAULT_CONFIG.copy()
cfg["discount_factor"] = 0.999
cfg["alpha"] = 0.4
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = "auto" if not args.eval else 0
cfg["experiment"]["checkpoint_interval"] = "auto" if not args.eval else 0
cfg["experiment"]["directory"] = f"runs/torch/{task_name}"

agent = Q_LEARNING(
    models=models,
    memory=None,
    cfg=cfg,
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 80000, "headless": args.headless}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: '{args.checkpoint}'")
        exit(1)
    agent.load(args.checkpoint)

trainer.train() if not args.eval else trainer.eval()
