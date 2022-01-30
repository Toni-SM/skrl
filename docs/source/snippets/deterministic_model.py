
import torch
import torch.nn as nn

from skrl.models.torch import DeterministicModel


class Critic(DeterministicModel):
    def __init__(self, observation_space, action_space, device, clip_actions = False):
        super().__init__(observation_space, action_space, device, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 1))

    def compute(self, states, taken_actions):
        return self.net(torch.cat([states, taken_actions], dim=1))
