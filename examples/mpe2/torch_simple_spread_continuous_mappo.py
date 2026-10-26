import argparse
import os

import torch
import torch.nn as nn

from mpe2 import simple_spread_v3
from pettingzoo.utils import BaseParallelWrapper
from pettingzoo.utils.env_logger import EnvLogger

# import the skrl components to build the RL system
from skrl import logger
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_CFG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

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


# define the MLP (with layer normalization) used by the models
# - layers are initialized with orthogonal weights (ReLU gain for the hidden layers) and zero biases
def orthogonal_init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def mlp(input_size, output_size, hidden_sizes=(128, 128), output_gain=1.0):
    layers = [nn.LayerNorm(input_size)]
    for size in hidden_sizes:
        layers += [
            orthogonal_init(nn.Linear(input_size, size), gain=nn.init.calculate_gain("relu")),
            nn.ReLU(),
            nn.LayerNorm(size),
        ]
        input_size = size
    return nn.Sequential(*layers, orthogonal_init(nn.Linear(input_size, output_size), gain=output_gain))


# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2)

        self.net = mlp(self.num_observations, self.num_actions, output_gain=0.01)
        # standard deviation bounded to (0, 0.5) as 0.5 * sigmoid(parameter), with initial value 0.5 * sigmoid(1) ~ 0.37
        self.std_parameter = nn.Parameter(torch.ones(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {"log_std": torch.log(0.5 * torch.sigmoid(self.std_parameter))}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self)

        self.net = mlp(self.num_states, 1)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# define a PettingZoo wrapper to give each agent the team reward (the sum of the agents' rewards)
class TeamReward(BaseParallelWrapper):
    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        team_reward = sum(rewards.values())
        return observations, {uid: team_reward for uid in rewards}, terminations, truncations, infos


# the policy's actions are not clipped (the environment clips them): suppress the environment's warnings about it
EnvLogger.suppress_output()

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


# instantiate memories as rollout buffer (any memory can be used for this)
memories = {}
for agent_name in env.possible_agents:
    memories[agent_name] = RandomMemory(memory_size=200, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# MAPPO requires 2 models per agent, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/multi_agents/mappo.html#models
models = {}
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["policy"] = Policy(
        env.observation_space(agent_name), env.state_space(agent_name), env.action_space(agent_name), device
    )
    models[agent_name]["value"] = Value(
        env.observation_space(agent_name), env.state_space(agent_name), env.action_space(agent_name), device
    )


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/multi_agents/mappo.html#configuration-and-hyperparameters
cfg = MAPPO_CFG()
cfg.rollouts = 200  # memory_size
cfg.learning_epochs = 5
cfg.mini_batches = 1
cfg.discount_factor = 0.99
cfg.gae_lambda = 0.95
cfg.learning_rate = 5e-4
cfg.value_preprocessor = RunningStandardScaler
cfg.value_preprocessor_kwargs = {"size": 1, "device": device}
cfg.grad_norm_clip = 10.0
cfg.ratio_clip = 0.2
cfg.value_clip = 0.2
cfg.entropy_loss_scale = 0.01
cfg.value_loss_scale = 1.0
cfg.value_loss_type = "huber"
cfg.huber_delta = 10.0
cfg.kl_threshold = 0
cfg.time_limit_bootstrap = True
cfg.mixed_precision = False
# logging to TensorBoard and write checkpoints (in timesteps)
cfg.experiment.write_interval = "auto" if not args.eval else 0
cfg.experiment.checkpoint_interval = "auto" if not args.eval else 0
cfg.experiment.directory = f"runs/torch/{task_name}"

agent = MAPPO(
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
