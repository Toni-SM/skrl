import gym

class DummyEnv:
    observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
    action_space = gym.spaces.Discrete(2)
    device = "cuda:0"

env = DummyEnv()

# [start-mlp]
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import CategoricalModel


# define the model
class MLP(CategoricalModel):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        super().__init__(observation_space, action_space, device, unnormalized_log_prob)

        self.linear_layer_1 = nn.Linear(self.num_observations, 64)
        self.linear_layer_2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, self.num_actions)

    def compute(self, states, taken_actions):
        x = F.relu(self.linear_layer_1(states))
        x = F.relu(self.linear_layer_2(x))
        return self.output_layer(x)


# instantiate the model (assumes there is a wrapped environment: env)
policy = MLP(observation_space=env.observation_space, 
             action_space=env.action_space, 
             device=env.device, 
             unnormalized_log_prob=True)
# [end-mlp]

import torch
policy.to(env.device)
actions = policy.act(torch.randn(10, 4, device=env.device))
assert actions[0].shape == torch.Size([10, 1])

# =============================================================================

import gym

class DummyEnv:
    observation_space = gym.spaces.Box(low=0, high=255, shape=(128, 128, 3))
    action_space = gym.spaces.Discrete(3)
    device = "cuda:0"

env = DummyEnv()

# [start-cnn]
import torch.nn as nn

from skrl.models.torch import CategoricalModel


# define the model
class CNN(CategoricalModel):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        super().__init__(observation_space, action_space, device, unnormalized_log_prob)

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

    def compute(self, states, taken_actions):
        # permute (samples, width, height, channels) -> (samples, channels, width, height) 
        return self.net(states.permute(0, 3, 1, 2))


# instantiate the model (assumes there is a wrapped environment: env)
policy = CNN(observation_space=env.observation_space, 
             action_space=env.action_space, 
             device=env.device, 
             unnormalized_log_prob=True)
# [end-cnn]

import torch
policy.to(env.device)
actions = policy.act(torch.randn(10, 128, 128, 3, device=env.device))
assert actions[0].shape == torch.Size([10, 1])
