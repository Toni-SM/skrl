# [start-mlp]
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import Model, CategoricalMixin


# define the model
class MLP(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.linear_layer_1 = nn.Linear(self.num_observations, 64)
        self.linear_layer_2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, self.num_actions)

    def compute(self, states, taken_actions, role):
        x = F.relu(self.linear_layer_1(states))
        x = F.relu(self.linear_layer_2(x))
        return self.output_layer(x)


# instantiate the model (assumes there is a wrapped environment: env)
policy = MLP(observation_space=env.observation_space, 
             action_space=env.action_space, 
             device=env.device, 
             unnormalized_log_prob=True)
# [end-mlp]

# =============================================================================

# [start-cnn]
import torch.nn as nn

from skrl.models.torch import Model, CategoricalMixin


# define the model
class CNN(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

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

    def compute(self, states, taken_actions, role):
        # permute (samples, width, height, channels) -> (samples, channels, width, height) 
        return self.net(states.permute(0, 3, 1, 2))


# instantiate the model (assumes there is a wrapped environment: env)
policy = CNN(observation_space=env.observation_space, 
             action_space=env.action_space, 
             device=env.device, 
             unnormalized_log_prob=True)
# [end-cnn]
