import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F

# import the skrl components to build the RL system
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


TRAIN = True

# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define models (deterministic models) using mixin
class Actor(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, self.num_actions),
                                 nn.Tanh())

    def compute(self, inputs, role):
        x = self.net(inputs["observations"])
        return 2 * x, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 1))

    def compute(self, inputs, role):
        x = self.net(torch.cat([inputs["observations"], inputs["taken_actions"]], dim=1))
        return x, {}


# load and wrap the gymnasium environment.
# note: the environment version may change depending on the gymnasium version
try:
    env = gym.make("Pendulum-v1", render_mode="human" if not TRAIN else None)
except (gym.error.DeprecatedEnv, gym.error.VersionNotFound) as e:
    env_id = [spec for spec in gym.envs.registry if spec.startswith("Pendulum-v")][0]
    print("Pendulum-v1 not found. Trying {}".format(env_id))
    env = gym.make(env_id)
env = wrap_env(env)

device = env.device


# instantiate a memory as experience replay
memory = RandomMemory(memory_size=15000, num_envs=env.num_envs, device=device, replacement=False)


# instantiate the agent's models (function approximators).
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#models
models = {}
models["policy"] = Actor(env.observation_space, env.state_space, env.action_space, device)
models["target_policy"] = Actor(env.observation_space, env.state_space, env.action_space, device)
models["critic"] = Critic(env.observation_space, env.state_space, env.action_space, device)
models["target_critic"] = Critic(env.observation_space, env.state_space, env.action_space, device)

# # initialize models' parameters (weights and biases)
# for model in models.values():
#     model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#configuration-and-hyperparameters
cfg = DDPG_DEFAULT_CONFIG.copy()
cfg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=1.0, device=device)
cfg["batch_size"] = 100
cfg["random_timesteps"] = 100
cfg["learning_starts"] = 100
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 75 if TRAIN else 0
cfg["experiment"]["checkpoint_interval"] = 750 if TRAIN else 0
cfg["experiment"]["directory"] = "runs/torch/Pendulum"

agent = DDPG(models=models,
             memory=memory,
             cfg=cfg,
             observation_space=env.observation_space,
             state_space=env.state_space,
             action_space=env.action_space,
             device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 15000, "headless": TRAIN}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

if TRAIN:
    # start training
    trainer.train()
else:
    # start evaluation
    # agent.load("/home/toni/Documents/RL/skrl-org/runs/torch/Pendulum/25-06-17_17-00-34-236778_DDPG/checkpoints/agent_15000.pt")
    agent.load("/home/toni/Documents/RL/skrl-org/runs/torch/Pendulum/25-06-17_17-04-29-378954_DDPG/checkpoints/agent_15000.pt")
    trainer.eval()
