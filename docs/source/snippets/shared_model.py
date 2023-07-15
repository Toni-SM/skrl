# [start-mlp-torch]
import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


# define the shared model
class SharedModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction, role="policy")
        DeterministicMixin.__init__(self, clip_actions, role="value")

        # shared layers/network
        self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 32),
                                 nn.ELU())

        # separated layers ("policy")
        self.mean_layer = nn.Linear(32, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        # separated layer ("value")
        self.value_layer = nn.Linear(32, 1)

    # override the .act(...) method to disambiguate its call
    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    # forward the input to compute model output according to the specified role
    def compute(self, inputs, role):
        if role == "policy":
            return self.mean_layer(self.net(inputs["states"])), self.log_std_parameter, {}
        elif role == "value":
            return self.value_layer(self.net(inputs["states"])), {}


# instantiate the shared model and pass the same instance to the other key
models = {}
models["policy"] = SharedModel(env.observation_space, env.action_space, env.device)
models["value"] = models["policy"]
# [end-mlp-torch]
