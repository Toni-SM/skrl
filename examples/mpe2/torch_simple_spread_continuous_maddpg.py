import argparse
import os

import torch
import torch.nn as nn

from mpe2 import simple_spread_v3
from pettingzoo.utils import BaseParallelWrapper

# import the skrl components to build the RL system
from skrl import logger
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.multi_agents.torch.maddpg import MADDPG, MADDPG_CFG
from skrl.resources.noises.torch import GaussianNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.spaces.torch import compute_joint_space

# import the vectorized MPE2 environment (mpe2_vectorized_env.py, in this directory)
from mpe2_vectorized_env import VectorizedParallelEnv


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=20, help="Number of environments")
parser.add_argument("--num_workers", type=int, default=4, help="Number of environment worker processes")
parser.add_argument("--timesteps", type=int, default=500000, help="Number of timesteps (steps of all environments)")
parser.add_argument("--headless", action="store_true", help="Run in headless mode (no rendering)")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (logging/checkpointing disabled)")
args, _ = parser.parse_known_args()


# seed for reproducibility
set_seed(args.seed)  # e.g. `set_seed(42)` for fixed seed


# define the MLP used by the models
def mlp(input_size, output_size, hidden_sizes=(128, 128)):
    layers = []
    for size in hidden_sizes:
        layers += [nn.Linear(input_size, size), nn.ReLU()]
        input_size = size
    return nn.Sequential(*layers, nn.Linear(input_size, output_size))


# define models (deterministic models) using mixins
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=True)

        self.net = nn.Sequential(mlp(self.num_observations, self.num_actions), nn.Tanh())

    def compute(self, inputs, role):
        # scale the actions from [-1, 1] (tanh) to the environment's action space [0, 1]
        return 0.5 * (self.net(inputs["observations"]) + 1), {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self)

        self.net = mlp(self.num_states + self.num_actions, 1)

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


# define a PettingZoo wrapper to give each agent the team reward (the sum of the agents' rewards)
class TeamReward(BaseParallelWrapper):
    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        team_reward = sum(rewards.values())
        return observations, {uid: team_reward for uid in rewards}, terminations, truncations, infos


# load the environment (vectorized MPE2 environment whose outputs are batched per agent)
task_name = "simple_spread_continuous"
env = VectorizedParallelEnv(
    lambda i: TeamReward(
        simple_spread_v3.parallel_env(
            continuous_actions=True,
            max_cycles=25,
            render_mode="human" if not args.headless and i == 0 else None,
        )
    ),
    num_envs=args.num_envs,
    num_workers=args.num_workers,
)
# wrap the environment
env = wrap_env(env, wrapper="pettingzoo")

device = env.device


# instantiate a replay memory per agent (the same transitions are sampled from all the memories,
# so they must have the same size and number of environments)
memories = {}
for agent_name in env.possible_agents:
    memories[agent_name] = RandomMemory(memory_size=50000, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# MADDPG requires 4 models per agent, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/multi_agents/maddpg.html#models
# - the critics (which take the environment state and the joint action of all agents) are shared by all agents:
#   the same instances are used for each. They are built with the joint action space of all agents
joint_action_space = compute_joint_space(
    [env.action_space(agent_name) for agent_name in env.possible_agents], mode="concatenate"
)
critic = Critic(
    env.observation_space(env.possible_agents[0]), env.state_space(env.possible_agents[0]), joint_action_space, device
)
target_critic = Critic(
    env.observation_space(env.possible_agents[0]), env.state_space(env.possible_agents[0]), joint_action_space, device
)
models = {}
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["policy"] = DeterministicActor(
        env.observation_space(agent_name), env.state_space(agent_name), env.action_space(agent_name), device
    )
    models[agent_name]["target_policy"] = DeterministicActor(
        env.observation_space(agent_name), env.state_space(agent_name), env.action_space(agent_name), device
    )
    models[agent_name]["critic"] = critic
    models[agent_name]["target_critic"] = target_critic


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/multi_agents/maddpg.html#configuration-and-hyperparameters
cfg = MADDPG_CFG()
cfg.exploration_noise = GaussianNoise
cfg.exploration_noise_kwargs = {"mean": 0, "std": 0.05, "device": device}
cfg.gradient_steps = 1
cfg.batch_size = 1000
cfg.discount_factor = 0.99
cfg.polyak = 0.005
cfg.learning_rate = (5e-4, 1e-3)  # (policy, critic)
cfg.random_timesteps = 500
cfg.learning_starts = 500
cfg.mixed_precision = False
# logging to TensorBoard and write checkpoints (in timesteps)
cfg.experiment.write_interval = "auto" if not args.eval else 0
cfg.experiment.checkpoint_interval = "auto" if not args.eval else 0
cfg.experiment.directory = f"runs/torch/{task_name}"

agent = MADDPG(
    possible_agents=env.possible_agents,
    models=models,
    memories=memories,
    cfg=cfg,
    observation_spaces=env.observation_spaces,
    state_spaces=env.state_spaces,
    action_spaces=env.action_spaces,
    device=device,
)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": args.timesteps // env.num_envs, "headless": args.headless}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: '{args.checkpoint}'")
        exit(1)
    agent.load(args.checkpoint)

trainer.train() if not args.eval else trainer.eval()
