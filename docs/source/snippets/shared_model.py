# [start-mlp-single-forward-pass-torch]
import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


# define the shared model
class SharedModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        clip_actions=False,
        clip_mean_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_mean_actions=clip_mean_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
            role="policy",
        )
        DeterministicMixin.__init__(self, clip_actions=False, role="value")

        # shared layers/network
        self.net = nn.Sequential(nn.Linear(self.num_observations, 32), nn.ELU(), nn.Linear(32, 32), nn.ELU())

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
            # save shared layers/network output to perform a single forward-pass
            self._shared_output = self.net(inputs["observations"])
            return self.mean_layer(self._shared_output), {"log_std": self.log_std_parameter}
        elif role == "value":
            # use saved shared layers/network output to perform a single forward-pass, if it was saved
            shared_output = self.net(inputs["observations"]) if self._shared_output is None else self._shared_output
            self._shared_output = (
                None  # reset saved shared output to prevent the use of erroneous data in subsequent steps
            )
            return self.value_layer(shared_output), {}


# instantiate the shared model and pass the same instance to the other key
models = {}
models["policy"] = SharedModel(env.observation_space, env.state_space, env.action_space, env.device)
models["value"] = models["policy"]
# [end-mlp-single-forward-pass-torch]


# [start-mlp-multi-forward-pass-torch]
import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


# define the shared model
class SharedModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        clip_actions=False,
        clip_mean_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_mean_actions=clip_mean_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
            role="policy",
        )
        DeterministicMixin.__init__(self, clip_actions=False, role="value")

        # shared layers/network
        self.net = nn.Sequential(nn.Linear(self.num_observations, 32), nn.ELU(), nn.Linear(32, 32), nn.ELU())

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
            return self.mean_layer(self.net(inputs["observations"])), {"log_std": self.log_std_parameter}
        elif role == "value":
            return self.value_layer(self.net(inputs["observations"])), {}


# instantiate the shared model and pass the same instance to the other key
models = {}
models["policy"] = SharedModel(env.observation_space, env.state_space, env.action_space, env.device)
models["value"] = models["policy"]
# [end-mlp-multi-forward-pass-torch]
