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
from skrl.multi_agents.torch.had3qn import HAD3QN, HAD3QN_CFG
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


# define models (deterministic models) using mixins
class DuelingQNetwork(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, use_states=False):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self)

        self.use_states = use_states
        self.base = nn.Sequential(
            nn.Linear(self.num_states if use_states else self.num_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(nn.Linear(128, 128), nn.Hardswish(), nn.Linear(128, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(128, 128), nn.Hardswish(), nn.Linear(128, self.num_actions))

    def compute(self, inputs, role):
        features = self.base(inputs["states"] if self.use_states else inputs["observations"])
        advantages = self.advantage_stream(features)
        return self.value_stream(features) + advantages - advantages.mean(dim=-1, keepdim=True), {}


# define a PettingZoo wrapper to give each agent the team reward (the sum of the agents' rewards)
class TeamReward(BaseParallelWrapper):
    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        team_reward = sum(rewards.values())
        return observations, {uid: team_reward for uid in rewards}, terminations, truncations, infos


# load the environment (vectorized MPE2 environment whose outputs are batched per agent)
task_name = "simple_spread_discrete"
env = VectorizedParallelEnv(
    lambda i: TeamReward(
        simple_spread_v3.parallel_env(
            continuous_actions=False,
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
# HAD3QN requires 4 models per agent, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/multi_agents/had3qn.html#models
# - the critics (which take the environment state and the joint action of all agents) are shared by all agents:
#   the same instances are used for each. They are built with the joint action space of all agents
joint_action_space = compute_joint_space(
    [env.action_space(agent_name) for agent_name in env.possible_agents], mode="product"
)
critic = DuelingQNetwork(
    env.observation_space(env.possible_agents[0]),
    env.state_space(env.possible_agents[0]),
    joint_action_space,
    device,
    use_states=True,
)
target_critic = DuelingQNetwork(
    env.observation_space(env.possible_agents[0]),
    env.state_space(env.possible_agents[0]),
    joint_action_space,
    device,
    use_states=True,
)
models = {}
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["q_network"] = DuelingQNetwork(
        env.observation_space(agent_name), env.state_space(agent_name), env.action_space(agent_name), device
    )
    models[agent_name]["target_q_network"] = DuelingQNetwork(
        env.observation_space(agent_name), env.state_space(agent_name), env.action_space(agent_name), device
    )
    models[agent_name]["critic"] = critic
    models[agent_name]["target_critic"] = target_critic


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/multi_agents/had3qn.html#configuration-and-hyperparameters
cfg = HAD3QN_CFG()
cfg.gradient_steps = 1
cfg.batch_size = 1000
cfg.discount_factor = 0.95
cfg.polyak = 0.005
cfg.learning_rate = (5e-4, 1e-3)  # (Q-networks, critic)
cfg.random_timesteps = 500
cfg.learning_starts = 500
cfg.update_interval = 1
cfg.target_update_interval = 1
cfg.exploration_scheduler = lambda timestep, timesteps: 0.05  # constant epsilon
cfg.mixed_precision = False
cfg.fixed_order = False
# logging to TensorBoard and write checkpoints (in timesteps)
cfg.experiment.write_interval = "auto" if not args.eval else 0
cfg.experiment.checkpoint_interval = "auto" if not args.eval else 0
cfg.experiment.directory = f"runs/torch/{task_name}"

agent = HAD3QN(
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
