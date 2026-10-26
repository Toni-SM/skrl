import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mpe2 import simple_spread_v3
from pettingzoo.utils import BaseParallelWrapper

# import the skrl components to build the RL system
from skrl import logger
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.multi_agents.torch.hasac import HASAC, HASAC_CFG
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
def mlp(input_size, output_size, hidden_sizes=(256, 256)):
    layers = []
    for size in hidden_sizes:
        layers += [nn.Linear(input_size, size), nn.ReLU()]
        input_size = size
    return nn.Sequential(*layers, nn.Linear(input_size, output_size))


# define models (stochastic and deterministic models)
# - tanh-squashed Gaussian policy with state-dependent standard deviation: the actions are sampled from a Gaussian
#   distribution, squashed with tanh and scaled to the action space, and their log-probability is corrected accordingly
class SquashedGaussianActor(Model):
    def __init__(self, observation_space, state_space, action_space, device, min_log_std=-20, max_log_std=2):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        # the last hidden layer uses tanh to bound the features of the mean/log_std heads
        self.net = nn.Sequential(nn.Linear(self.num_observations, 256), nn.ReLU(), nn.Linear(256, 256), nn.Tanh())
        self.mean_layer = nn.Linear(256, self.num_actions)
        self.log_std_layer = nn.Linear(256, self.num_actions)
        # scale and bias to map the squashed actions from [-1, 1] to the action space
        low = torch.tensor(action_space.low, device=self.device)
        high = torch.tensor(action_space.high, device=self.device)
        self.register_buffer("action_scale", (high - low) / 2)
        self.register_buffer("action_bias", (high + low) / 2)

    def compute(self, inputs, role):
        features = self.net(inputs["observations"])
        log_std = torch.clamp(self.log_std_layer(features), self.min_log_std, self.max_log_std)
        return self.mean_layer(features), {"log_std": log_std}

    def act(self, inputs, *, role=""):
        mean, outputs = self.compute(inputs, role)
        log_std = outputs["log_std"]
        distribution = torch.distributions.Normal(mean, log_std.exp())
        # sample (using the reparameterization trick) and squash the actions
        samples = distribution.rsample()
        actions = torch.tanh(samples) * self.action_scale + self.action_bias
        # log-probability with the tanh (numerically stable form) and scaling corrections
        log_prob = distribution.log_prob(samples) - 2 * (math.log(2) - samples - F.softplus(-2 * samples))
        log_prob = (log_prob - torch.log(self.action_scale)).sum(dim=-1, keepdim=True)
        mean_actions = torch.tanh(mean) * self.action_scale + self.action_bias
        return actions, {"log_prob": log_prob, "mean_actions": mean_actions}


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
# HASAC requires 5 models per agent, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/multi_agents/hasac.html#models
# - the critics (which take the environment state and the joint action of all agents) are shared by all agents:
#   the same instances are used for each. They are built with the joint action space of all agents
joint_action_space = compute_joint_space(
    [env.action_space(agent_name) for agent_name in env.possible_agents], mode="concatenate"
)
critic_1 = Critic(
    env.observation_space(env.possible_agents[0]), env.state_space(env.possible_agents[0]), joint_action_space, device
)
critic_2 = Critic(
    env.observation_space(env.possible_agents[0]), env.state_space(env.possible_agents[0]), joint_action_space, device
)
target_critic_1 = Critic(
    env.observation_space(env.possible_agents[0]), env.state_space(env.possible_agents[0]), joint_action_space, device
)
target_critic_2 = Critic(
    env.observation_space(env.possible_agents[0]), env.state_space(env.possible_agents[0]), joint_action_space, device
)
models = {}
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["policy"] = SquashedGaussianActor(
        env.observation_space(agent_name), env.state_space(agent_name), env.action_space(agent_name), device
    )
    models[agent_name]["critic_1"] = critic_1
    models[agent_name]["critic_2"] = critic_2
    models[agent_name]["target_critic_1"] = target_critic_1
    models[agent_name]["target_critic_2"] = target_critic_2


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/multi_agents/hasac.html#configuration-and-hyperparameters
cfg = HASAC_CFG()
cfg.gradient_steps = 1
cfg.batch_size = 1000
cfg.discount_factor = 0.99
cfg.polyak = 0.005
cfg.learning_rate = (5e-4, 5e-4, 3e-4)  # (policy, critic, entropy coefficient)
cfg.random_timesteps = 500
cfg.learning_starts = 500
cfg.learn_entropy = True
cfg.initial_entropy_value = 1.0
cfg.mixed_precision = False
cfg.fixed_order = False
# logging to TensorBoard and write checkpoints (in timesteps)
cfg.experiment.write_interval = "auto" if not args.eval else 0
cfg.experiment.checkpoint_interval = "auto" if not args.eval else 0
cfg.experiment.directory = f"runs/torch/{task_name}"

agent = HASAC(
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
