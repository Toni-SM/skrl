import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import Model, DeterministicMixin, GaussianMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env


# Define the models (deterministic models) for the SAC agent using mixin
# - Actor (policy): takes as input the environment's observation/state and returns an action
# - Critic: takes the state and action as input and provides a value to guide the policy
class Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.linear_layer_1 = nn.Linear(self.num_observations, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        # Pendulum-v1 action_space is -2 to 2
        return 2 * torch.tanh(self.action_layer(x)), self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, 1)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)))
        x = F.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x), {}


# Load and wrap the Gym environment.
# Note: the environment version may change depending on the gym version
try:
    env = gym.make("Pendulum-v1")
except gym.error.DeprecatedEnv as e:
    env_id = [spec.id for spec in gym.envs.registry.all() if spec.id.startswith("Pendulum-v")][0]
    print("Pendulum-v1 not found. Trying {}".format(env_id))
    env = gym.make(env_id)
env = wrap_env(env)

device = env.device


# Instantiate a RandomMemory (without replacement) as experience replay memory
memory = RandomMemory(memory_size=20000, num_envs=env.num_envs, device=device, replacement=False)


# Instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sac.html#spaces-and-models
models_sac = {}
models_sac["policy"] = Actor(env.observation_space, env.action_space, device, clip_actions=True)
models_sac["critic_1"] = Critic(env.observation_space, env.action_space, device)
models_sac["critic_2"] = Critic(env.observation_space, env.action_space, device)
models_sac["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models_sac["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models_sac.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sac.html#configuration-and-hyperparameters
cfg_sac = SAC_DEFAULT_CONFIG.copy()
cfg_sac["discount_factor"] = 0.98
cfg_sac["batch_size"] = 100
cfg_sac["random_timesteps"] = 0
cfg_sac["learning_starts"] = 1000
cfg_sac["learn_entropy"] = True
# logging to TensorBoard and write checkpoints each 75 and 750 timesteps respectively
cfg_sac["experiment"]["write_interval"] = 75
cfg_sac["experiment"]["checkpoint_interval"] = 750

agent_sac = SAC(models=models_sac,
                memory=memory,
                cfg=cfg_sac,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 15000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_sac)

# start training
trainer.train()
