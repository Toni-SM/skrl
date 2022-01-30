import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import CategoricalModel


class Policy(CategoricalModel):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        super().__init__(observation_space, action_space, device, unnormalized_log_prob)

        self.linear_layer_1 = nn.Linear(self.num_observations, 64)
        self.linear_layer_2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, self.num_actions)

    def compute(self, states, taken_actions):
        x = F.relu(self.linear_layer_1(states))
        x = F.relu(self.linear_layer_2(x))
        return self.output_layer(x)
