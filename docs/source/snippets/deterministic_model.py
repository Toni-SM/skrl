# [start-mlp]
import torch
import torch.nn as nn

from skrl.models.torch import Model, DeterministicMixin


# define the model
class MLP(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


# instantiate the model (assumes there is a wrapped environment: env)
policy = MLP(observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device,
             clip_actions=False)
# [end-mlp]

# =============================================================================

# [start-cnn]
import torch
import torch.nn as nn

from skrl.models.torch import Model, DeterministicMixin


# define the model
class CNN(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

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

    def compute(self, inputs, role):
        # permute (samples, width, height, channels) -> (samples, channels, width, height)
        x = self.features_extractor(inputs["states"].permute(0, 3, 1, 2))
        return self.net(torch.cat([x, inputs["taken_actions"]], dim=1)), {}


# instantiate the model (assumes there is a wrapped environment: env)
policy = CNN(observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device,
             clip_actions=False)
# [end-cnn]
