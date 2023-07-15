# [start-definition-torch]
class GaussianModel(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device=None,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
# [end-definition-torch]


# [start-definition-jax]
class GaussianModel(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device=None,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
# [end-definition-jax]

# =============================================================================

# [start-mlp-sequential-torch]
import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin


# define the model
class MLP(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, self.num_actions),
                                 nn.Tanh())

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


# instantiate the model (assumes there is a wrapped environment: env)
policy = MLP(observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device,
             clip_actions=True,
             clip_log_std=True,
             min_log_std=-20,
             max_log_std=2,
             reduction="sum")
# [end-mlp-sequential-torch]

# [start-mlp-functional-torch]
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import Model, GaussianMixin


# define the model
class MLP(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.fc1 = nn.Linear(self.num_observations, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = self.fc1(inputs["states"])
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return torch.tanh(x), self.log_std_parameter, {}


# instantiate the model (assumes there is a wrapped environment: env)
policy = MLP(observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device,
             clip_actions=True,
             clip_log_std=True,
             min_log_std=-20,
             max_log_std=2,
             reduction="sum")
# [end-mlp-functional-torch]

# [start-mlp-setup-jax]
import jax.numpy as jnp
import flax.linen as nn

from skrl.models.jax import Model, GaussianMixin


# define the model
class MLP(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device=None,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

    def setup(self):
        self.fc1 = nn.Dense(64)
        self.fc2 = nn.Dense(32)
        self.fc3 = nn.Dense(self.num_actions)

        self.log_std_parameter = self.param("log_std_parameter", lambda _: jnp.zeros(self.num_actions))

    def __call__(self, inputs, role):
        x = self.fc1(inputs["states"])
        x = nn.relu(x)
        x = self.fc2(x)
        x = nn.relu(x)
        x = self.fc3(x)
        return nn.tanh(x), self.log_std_parameter, {}


# instantiate the model (assumes there is a wrapped environment: env)
policy = MLP(observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device,
             clip_actions=True,
             clip_log_std=True,
             min_log_std=-20,
             max_log_std=2,
             reduction="sum")
# [end-mlp-setup-jax]

# [start-mlp-compact-jax]
import jax.numpy as jnp
import flax.linen as nn

from skrl.models.jax import Model, GaussianMixin


# define the model
class MLP(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device=None,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

    @nn.compact  # marks the given module method allowing inlined submodules
    def __call__(self, inputs, role):
        x = nn.Dense(64)(inputs["states"])
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        log_std_parameter = self.param("log_std_parameter", lambda _: jnp.zeros(self.num_actions))
        return nn.tanh(x), log_std_parameter, {}


# instantiate the model (assumes there is a wrapped environment: env)
policy = MLP(observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device,
             clip_actions=True,
             clip_log_std=True,
             min_log_std=-20,
             max_log_std=2,
             reduction="sum")
# [end-mlp-compact-jax]

# =============================================================================

# [start-cnn-sequential-torch]
import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin


# define the model
class CNN(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=4),
                                 nn.ReLU(),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 16),
                                 nn.Tanh(),
                                 nn.Linear(16, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, self.num_actions))

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # permute (samples, width * height * channels) -> (samples, channels, width, height)
        return self.net(inputs["states"].view(-1, *self.observation_space.shape).permute(0, 3, 1, 2)), self.log_std_parameter, {}


# instantiate the model (assumes there is a wrapped environment: env)
policy = CNN(observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device,
             clip_actions=True,
             clip_log_std=True,
             min_log_std=-20,
             max_log_std=2,
             reduction="sum")
# [end-cnn-sequential-torch]

# [start-cnn-functional-torch]
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import Model, GaussianMixin


# define the model
class CNN(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 16)
        self.fc3 = nn.Linear(16, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

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
        x = self.fc3(x)
        x = torch.tanh(x)
        x = self.fc4(x)
        x = torch.tanh(x)
        x = self.fc5(x)
        return x, self.log_std_parameter, {}


# instantiate the model (assumes there is a wrapped environment: env)
policy = CNN(observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device,
             clip_actions=True,
             clip_log_std=True,
             min_log_std=-20,
             max_log_std=2,
             reduction="sum")
# [end-cnn-functional-torch]

# =============================================================================

# [start-rnn-sequential-torch]
import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin


# define the model
class RNN(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum",
                 num_envs=1, num_layers=1, hidden_size=64, sequence_length=10):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hout
        self.sequence_length = sequence_length

        self.rnn = nn.RNN(input_size=self.num_observations,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(nn.Linear(self.hidden_size, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, self.num_actions),
                                 nn.Tanh())

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        # batch size (N) is the number of envs during rollout
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size)]}}  # hidden states (D ∗ num_layers, N, Hout)

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        # training
        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            # get the hidden states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.rnn(rnn_input[:,i0:i1,:], hidden_states)
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        return self.net(rnn_output), self.log_std_parameter, {"rnn": [hidden_states]}


# instantiate the model (assumes there is a wrapped environment: env)
policy = RNN(observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device,
             clip_actions=True,
             clip_log_std=True,
             min_log_std=-20,
             max_log_std=2,
             reduction="sum",
             num_envs=env.num_envs,
             num_layers=1,
             hidden_size=64,
             sequence_length=10)
# [end-rnn-sequential-torch]

# [start-rnn-functional-torch]
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import Model, GaussianMixin


# define the model
class RNN(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum",
                 num_envs=1, num_layers=1, hidden_size=64, sequence_length=10):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hout
        self.sequence_length = sequence_length

        self.rnn = nn.RNN(input_size=self.num_observations,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)  # batch_first -> (batch, sequence, features)

        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        # batch size (N) is the number of envs during rollout
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size)]}}  # hidden states (D ∗ num_layers, N, Hout)

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        # training
        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            # get the hidden states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.rnn(rnn_input[:,i0:i1,:], hidden_states)
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        x = self.fc1(rnn_output)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return torch.tanh(x), self.log_std_parameter, {"rnn": [hidden_states]}


# instantiate the model (assumes there is a wrapped environment: env)
policy = RNN(observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device,
             clip_actions=True,
             clip_log_std=True,
             min_log_std=-20,
             max_log_std=2,
             reduction="sum",
             num_envs=env.num_envs,
             num_layers=1,
             hidden_size=64,
             sequence_length=10)
# [end-rnn-functional-torch]

# =============================================================================

# [start-gru-sequential-torch]
import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin


# define the model
class GRU(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum",
                 num_envs=1, num_layers=1, hidden_size=64, sequence_length=10):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hout
        self.sequence_length = sequence_length

        self.gru = nn.GRU(input_size=self.num_observations,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(nn.Linear(self.hidden_size, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, self.num_actions),
                                 nn.Tanh())

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        # batch size (N) is the number of envs during rollout
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size)]}}  # hidden states (D ∗ num_layers, N, Hout)

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        # training
        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            # get the hidden states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.gru(rnn_input[:,i0:i1,:], hidden_states)
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, hidden_states = self.gru(rnn_input, hidden_states)
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, hidden_states = self.gru(rnn_input, hidden_states)

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        return self.net(rnn_output), self.log_std_parameter, {"rnn": [hidden_states]}


# instantiate the model (assumes there is a wrapped environment: env)
policy = GRU(observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device,
             clip_actions=True,
             clip_log_std=True,
             min_log_std=-20,
             max_log_std=2,
             reduction="sum",
             num_envs=env.num_envs,
             num_layers=1,
             hidden_size=64,
             sequence_length=10)
# [end-gru-sequential-torch]

# [start-gru-functional-torch]
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import Model, GaussianMixin


# define the model
class GRU(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum",
                 num_envs=1, num_layers=1, hidden_size=64, sequence_length=10):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hout
        self.sequence_length = sequence_length

        self.gru = nn.GRU(input_size=self.num_observations,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)  # batch_first -> (batch, sequence, features)

        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        # batch size (N) is the number of envs during rollout
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size)]}}  # hidden states (D ∗ num_layers, N, Hout)

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        # training
        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            # get the hidden states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.gru(rnn_input[:,i0:i1,:], hidden_states)
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, hidden_states = self.gru(rnn_input, hidden_states)
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, hidden_states = self.gru(rnn_input, hidden_states)

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        x = self.fc1(rnn_output)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return torch.tanh(x), self.log_std_parameter, {"rnn": [hidden_states]}


# instantiate the model (assumes there is a wrapped environment: env)
policy = GRU(observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device,
             clip_actions=True,
             clip_log_std=True,
             min_log_std=-20,
             max_log_std=2,
             reduction="sum",
             num_envs=env.num_envs,
             num_layers=1,
             hidden_size=64,
             sequence_length=10)
# [end-gru-functional-torch]

# =============================================================================

# [start-lstm-sequential-torch]
import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin


# define the model
class LSTM(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum",
                 num_envs=1, num_layers=1, hidden_size=64, sequence_length=10):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size=self.num_observations,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(nn.Linear(self.hidden_size, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, self.num_actions),
                                 nn.Tanh())

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        # batch size (N) is the number of envs during rollout
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        # training
        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    cell_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        return self.net(rnn_output), self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}


# instantiate the model (assumes there is a wrapped environment: env)
policy = LSTM(observation_space=env.observation_space,
              action_space=env.action_space,
              device=env.device,
              clip_actions=True,
              clip_log_std=True,
              min_log_std=-20,
              max_log_std=2,
              reduction="sum",
              num_envs=env.num_envs,
              num_layers=1,
              hidden_size=64,
              sequence_length=10)
# [end-lstm-sequential-torch]

# [start-lstm-functional-torch]
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import Model, GaussianMixin


# define the model
class LSTM(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum",
                 num_envs=1, num_layers=1, hidden_size=64, sequence_length=10):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size=self.num_observations,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)

        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        # batch size (N) is the number of envs during rollout
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        # training
        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    cell_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        x = self.fc1(rnn_output)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return torch.tanh(x), self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}


# instantiate the model (assumes there is a wrapped environment: env)
policy = LSTM(observation_space=env.observation_space,
              action_space=env.action_space,
              device=env.device,
              clip_actions=True,
              clip_log_std=True,
              min_log_std=-20,
              max_log_std=2,
              reduction="sum",
              num_envs=env.num_envs,
              num_layers=1,
              hidden_size=64,
              sequence_length=10)
# [end-lstm-functional-torch]
