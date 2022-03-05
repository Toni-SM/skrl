# import JetBot environment
from env import JetBotEnv

import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import DeterministicModel, GaussianModel
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env


# Define the models (stochastic and deterministic models) for the SAC agent using helper classes.
# - Actor (policy): takes as input the environment's observation/state and returns an action
# - Critic: takes the state and action as input and provides a value to guide the policy 
class Actor(GaussianModel):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        super().__init__(observation_space, action_space, device, clip_actions,
                         clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=4),
                                 nn.ReLU(),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(9216, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 16),
                                 nn.Tanh(),
                                 nn.Linear(16, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, states, taken_actions):
        # view (samples, width * height * channels) -> (samples, width, height, channels) 
        # permute (samples, width, height, channels) -> (samples, channels, width, height) 
        x = self.net(states.view(-1, *self.observation_space.shape).permute(0, 3, 1, 2))
        return 10 * torch.tanh(x), self.log_std_parameter   # JetBotEnv action_space is -10 to 10

class Critic(DeterministicModel):
    def __init__(self, observation_space, action_space, device, clip_actions = False):
        super().__init__(observation_space, action_space, device, clip_actions)

        self.features_extractor = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=4),
                                                nn.ReLU(),
                                                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                                nn.ReLU(),
                                                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                                nn.ReLU(),
                                                nn.Flatten(),
                                                nn.Linear(9216, 512),
                                                nn.ReLU(),
                                                nn.Linear(512, 16),
                                                nn.Tanh())
        self.net = nn.Sequential(nn.Linear(16 + self.num_actions, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, 1))

    def compute(self, states, taken_actions):
        # view (samples, width * height * channels) -> (samples, width, height, channels) 
        # permute (samples, width, height, channels) -> (samples, channels, width, height) 
        x = self.features_extractor(states.view(-1, *self.observation_space.shape).permute(0, 3, 1, 2))
        return self.net(torch.cat([x, taken_actions], dim=1))


# Load and wrap the JetBot environment (a subclass of Gym)
env = JetBotEnv(headless=True)
env = wrap_env(env)

device = env.device


# Instanciate a RandomMemory (without replacement) as experience replay memory
memory = RandomMemory(memory_size=10000, num_envs=env.num_envs, device=device, replacement=False)


# Instanciate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sac.html#models-networks
models_sac = {"policy": Actor(env.observation_space, env.action_space, device, clip_actions=True),
                "critic_1": Critic(env.observation_space, env.action_space, device),
                "critic_2": Critic(env.observation_space, env.action_space, device),
                "target_critic_1": Critic(env.observation_space, env.action_space, device),
                "target_critic_2": Critic(env.observation_space, env.action_space, device)}

# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models_sac.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# Configure and instanciate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sac.html#configuration-and-hyperparameters
cfg_sac = SAC_DEFAULT_CONFIG.copy()
cfg_sac["gradient_steps"] = 1
cfg_sac["batch_size"] = 512
cfg_sac["random_timesteps"] = 0
cfg_sac["learning_starts"] = 1000
cfg_sac["learn_entropy"] = True
# logging to TensorBoard and write checkpoints each 1000 and 50000 timesteps respectively
cfg_sac["experiment"]["write_interval"] = 1000
cfg_sac["experiment"]["checkpoint_interval"] = 50000

agent = SAC(models=models_sac, 
            memory=memory, 
            cfg=cfg_sac, 
            observation_space=env.observation_space, 
            action_space=env.action_space,
            device=device)


# Configure and instanciate the RL trainer
cfg_trainer = {"timesteps": 500000, "headless": not True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()


# close the environment
env.close()
