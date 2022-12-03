# [start-mlp-sequential]
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
# [end-mlp-sequential]

# [start-mlp-functional]
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import Model, DeterministicMixin


# define the model
class MLP(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.fc1 = nn.Linear(self.num_observations + self.num_actions, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def compute(self, inputs, role):
        x = self.fc1(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1))
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x), {}


# instantiate the model (assumes there is a wrapped environment: env)
policy = MLP(observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device,
             clip_actions=False)
# [end-mlp-functional]

# =============================================================================

# [start-cnn-sequential]
import torch
import torch.nn as nn

from skrl.models.torch import Model, DeterministicMixin


# define the model
class CNN(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.features_extractor = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=4),
                                                nn.ReLU(),
                                                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                                nn.ReLU(),
                                                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                                nn.ReLU(),
                                                nn.Flatten(),
                                                nn.Linear(1024, 512),
                                                nn.ReLU(),
                                                nn.Linear(512, 16),
                                                nn.Tanh())

        self.net = nn.Sequential(nn.Linear(16 + self.num_actions, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, 1))

    def compute(self, inputs, role):
        # permute (samples, width * height * channels) -> (samples, channels, width, height)
        x = self.features_extractor(inputs["states"].view(-1, *self.observation_space.shape).permute(0, 3, 1, 2))
        return self.net(torch.cat([x, inputs["taken_actions"]], dim=1)), {}


# instantiate the model (assumes there is a wrapped environment: env)
policy = CNN(observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device,
             clip_actions=False)
# [end-cnn-sequential]

# [start-cnn-functional]
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import Model, DeterministicMixin


# define the model
class CNN(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 16)
        self.fc3 = nn.Linear(16 + self.num_actions, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

    def compute(self, inputs, role):
        # permute (samples, width * height * channels) -> (samples, channels, width, height)
        x = inputs["states"].view(-1, *self.observation_space.shape).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(torch.cat([x, inputs["taken_actions"]], dim=1))
        x = torch.tanh(x)
        x = self.fc4(x)
        x = torch.tanh(x)
        x = self.fc5(x)
        return x, {}


# instantiate the model (assumes there is a wrapped environment: env)
policy = CNN(observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device,
             clip_actions=False)
# [end-cnn-functional]
