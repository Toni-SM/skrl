import argparse
import os

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl import logger
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.multi_agents.torch.maddpg import MADDPG, MADDPG_CFG
from skrl.resources.noises.torch import GaussianNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.spaces.torch import compute_joint_space


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (logging/checkpointing disabled)")


# define models (deterministic models) using mixins
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, clip_actions=False):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions),
            nn.Tanh(),
        )

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


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

        self.net = nn.Sequential(
            nn.Linear(self.num_states + self.num_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


# load the environment
task_name = "Isaac-Pendulum-MARL-Direct"
env = load_isaaclab_env(task_name=task_name, parser=parser, num_envs=64)
# wrap the environment
env = wrap_env(env)

device = env.device


# defer parsing of arguments to include loader arguments (run with --help to see all the arguments)
args, _ = parser.parse_known_args()


# seed for reproducibility
set_seed(args.seed)  # e.g. `set_seed(42)` for fixed seed


# instantiate a replay memory per agent (the same transitions are sampled from all the memories,
# so they must have the same size and number of environments)
memories = {}
for agent_name in env.possible_agents:
    memories[agent_name] = RandomMemory(memory_size=16000, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# MADDPG requires 4 models per agent, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/multi_agents/maddpg.html#models
# - the critic (which takes the environment state and the joint action of all agents) is shared by all agents:
#   the same instances are used for each. It is built with the joint action space of all agents
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
cfg.exploration_noise_kwargs = {"mean": 0, "std": 0.1, "device": device}
cfg.gradient_steps = 1
cfg.batch_size = 4096
cfg.discount_factor = 0.99
cfg.polyak = 0.005
cfg.learning_rate = 5e-4
cfg.random_timesteps = 50
cfg.learning_starts = 50
cfg.observation_preprocessor = RunningStandardScaler
cfg.observation_preprocessor_kwargs = {
    agent_name: {"size": env.observation_space(agent_name), "device": device} for agent_name in env.possible_agents
}
cfg.state_preprocessor = RunningStandardScaler
cfg.state_preprocessor_kwargs = {"size": env.state_space(env.possible_agents[0]), "device": device}
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
cfg_trainer = {"timesteps": 80000, "headless": not args.visualizer, "environment_info": "log"}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: '{args.checkpoint}'")
        exit(1)
    agent.load(args.checkpoint)

trainer.train() if not args.eval else trainer.eval()
