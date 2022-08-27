import gym

class DummyEnv:
    observation_space = gym.spaces.Box(low=-1, high=1, shape=(5,))
    action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
    device = "cuda:0"

env = DummyEnv()

# [start-mlp]
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import Model, MultivariateGaussianMixin


# define the model
class MLP(MultivariateGaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, 
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        MultivariateGaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.linear_layer_1 = nn.Linear(self.num_observations, 128)
        self.linear_layer_2 = nn.Linear(128, 64)
        self.linear_layer_3 = nn.Linear(64, 32)
        self.mean_action_layer = nn.Linear(32, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, states, taken_actions, role):
        x = F.relu(self.linear_layer_1(states))
        x = F.relu(self.linear_layer_2(x))
        x = F.relu(self.linear_layer_3(x))
        return torch.tanh(self.mean_action_layer(x)), self.log_std_parameter

# instantiate the model (assumes there is a wrapped environment: env)
policy = MLP(observation_space=env.observation_space, 
             action_space=env.action_space, 
             device=env.device, 
             clip_actions=True,
             clip_log_std=True,
             min_log_std=-20,
             max_log_std=2)
# [end-mlp]

# =============================================================================

# [start-cnn]
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import Model, MultivariateGaussianMixin


# define the model
class CNN(MultivariateGaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, 
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        MultivariateGaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Conv2d(1, 64, kernel_size=4, stride=2),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 32, kernel_size=4, stride=2),
                                 nn.ReLU(),
                                 nn.Conv2d(32, 16, kernel_size=2, stride=2),
                                 nn.ReLU(),
                                 nn.Conv2d(16, 8, kernel_size=2, stride=2),
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(1800, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 16),
                                 nn.Tanh(),
                                 nn.Linear(16, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, self.num_actions))
        
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, states, taken_actions, role):
        # permute (samples, width, height, channels) -> (samples, channels, width, height) 
        return self.net(states.permute(0, 3, 1, 2)), self.log_std_parameter


# instantiate the model (assumes there is a wrapped environment: env)
policy = CNN(observation_space=env.observation_space, 
             action_space=env.action_space, 
             device=env.device, 
             clip_actions=True,
             clip_log_std=True,
             min_log_std=-20,
             max_log_std=2)
# [end-cnn]
