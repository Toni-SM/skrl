import gym

class DummyEnv:
    observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
    action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
    device = "cuda:0"

env = DummyEnv()

# [start-mlp]
import torch
import torch.nn as nn

from skrl.models.torch import DeterministicModel


# define the model
class MLP(DeterministicModel):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        super().__init__(observation_space, action_space, device, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 1))

    def compute(self, states, taken_actions):
        return self.net(torch.cat([states, taken_actions], dim=1))


# instantiate the model (assumes there is a wrapped environment: env)
policy = MLP(observation_space=env.observation_space, 
             action_space=env.action_space, 
             device=env.device, 
             clip_actions=False)
# [end-mlp]

import torch
policy.to(env.device)
actions = policy.act(torch.randn(10, 4, device=env.device), torch.randn(10, 3, device=env.device))
assert actions[0].shape == torch.Size([10, 1])

# =============================================================================

import gym

class DummyEnv:
    observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3))
    action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
    device = "cuda:0"

env = DummyEnv()

# [start-cnn]
import torch
import torch.nn as nn

from skrl.models.torch import DeterministicModel


# define the model
class CNN(DeterministicModel):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        super().__init__(observation_space, action_space, device, clip_actions)

        self.features_extractor = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=3),
                                                nn.ReLU(),
                                                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                                nn.ReLU(),
                                                nn.Conv2d(64, 64, kernel_size=2, stride=1),
                                                nn.ReLU(),
                                                nn.Flatten(),
                                                nn.Linear(3136, 512),
                                                nn.ReLU(),
                                                nn.Linear(512, 16),
                                                nn.Tanh())
        self.net = nn.Sequential(nn.Linear(16 + self.num_actions, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, 1))

    def compute(self, states, taken_actions):
        # permute (samples, width, height, channels) -> (samples, channels, width, height) 
        x = self.features_extractor(states.permute(0, 3, 1, 2))
        return self.net(torch.cat([x, taken_actions], dim=1))


# instantiate the model (assumes there is a wrapped environment: env)
policy = CNN(observation_space=env.observation_space, 
             action_space=env.action_space, 
             device=env.device, 
             clip_actions=False)
# [end-cnn]

import torch
policy.to(env.device)
actions = policy.act(torch.randn(10, 64, 64, 3, device=env.device), torch.randn(10, 3, device=env.device))
assert actions[0].shape == torch.Size([10, 1])
