from typing import Any, Mapping, Tuple, Union

import gymnasium

import torch
from torch.distributions import Beta


# speed up distribution construction by disabling checking
Beta.set_default_validate_args(False)
EPS = 1e-6

class BetaMixin:
    def __init__(
        self,
        reduction: str = "sum",
        role: str = "",
    ) -> None:
        """Beta mixin model (stochastic model)

        :param reduction: Reduction method for returning the log probability density function: (default: ``"sum"``).
                          Supported values are ``"mean"``, ``"sum"``, ``"prod"`` and ``"none"``. If "``none"``, the log probability density
                          function is returned as a tensor of shape ``(num_samples, num_actions)`` instead of ``(num_samples, 1)``
        :type reduction: str, optional
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :raises ValueError: If the reduction method is not valid

        Example::

            # define the model
            >>> import torch
            >>> import torch.nn as nn
            >>> from skrl.models.torch import Model, BetaMixin
            >>>
            >>> class Policy(BetaMixin, Model):
            ...     def __init__(self, observation_space, action_space, device="cuda:0", reduction="sum"):
            ...         Model.__init__(self, observation_space, action_space, device)
            ...         BetaMixin.__init__(self, reduction)
            ...
            ...         self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
            ...                                  nn.ELU(),
            ...                                  nn.Linear(32, 32),
            ...                                  nn.ELU(),
            ...                                  nn.Linear(32, self.num_actions))
            ...         self.alpha = nn.Linear(32, self.num_actions)
            ...         self.beta = nn.Linear(32, self.num_actions)
            ...         self.alpha_activation = nn.Softplus()
            ...         self.beta_activation = nn.Softplus()
            ...
            ...     def compute(self, inputs, role):
            ...         alpha = self.alpha_activation(self.alpha(self.net(inputs["states"]))) + 1
            ...         beta = self.beta_activation(self.beta(self.net(inputs["states"]))) + 1
            ...         return alpha, beta, {"mean_actions": None}
            ...
            >>> # given an observation_space: gymnasium.spaces.Box with shape (60,)
            >>> # and an action_space: gymnasium.spaces.Box with shape (8,)
            >>> model = Policy(observation_space, action_space)
            >>>
            >>> print(model)
            Policy(
              (net): Sequential(
                (0): Linear(in_features=60, out_features=32, bias=True)
                (1): ELU(alpha=1.0)
                (2): Linear(in_features=32, out_features=32, bias=True)
                (3): ELU(alpha=1.0)
                (4): Linear(in_features=32, out_features=8, bias=True)
              )
              (alpha): Linear(in_features=32, out_features=8, bias=True)
              (beta): Linear(in_features=32, out_features=8, bias=True)
              (alpha_activation): Softplus(beta=1, threshold=20)
              (beta_activation): Softplus(beta=1, threshold=20)
            )
        """

        # Preven infinity values in action space and replace them with -1.0 and 1.0
        for i, _ in enumerate(self.action_space.low):
            if self.action_space.low[i] == -float("inf"):
                self.action_space.low[i] = -1.0
        for i, _ in enumerate(self.action_space.high):
            if self.action_space.high[i] == float("inf"):
                self.action_space.high[i] = 1.0

        self._b_actions_min = torch.tensor(self.action_space.low, device=self.device, dtype=torch.float32)
        self._b_actions_max = torch.tensor(self.action_space.high, device=self.device, dtype=torch.float32)

        self._b_log_std = None
        self._b_num_samples = None
        self._b_distribution = None

        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._b_reduction = (
            torch.mean
            if reduction == "mean"
            else torch.sum if reduction == "sum" else torch.prod if reduction == "prod" else None
        )

    def act(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act stochastically in response to the state of the environment

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function.
                 The third component is a dictionary containing the mean actions ``"mean_actions"``
                 and extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dict

        Example::

            >>> # given a batch of sample states with shape (4096, 60)
            >>> actions, log_prob, outputs = model.act({"states": states})
            >>> print(actions.shape, log_prob.shape, outputs["mean_actions"].shape)
            torch.Size([4096, 8]) torch.Size([4096, 1]) torch.Size([4096, 8])
        """
        # map from states/observations to mean actions and log standard deviations
        a, b, outputs = self.compute(inputs, role)
        self._b_num_samples = a.shape[0]

        # distribution
        self._b_distribution = Beta(a, b)
        self._b_log_std = torch.sqrt(a * b / ((a + b + 1) * (a + b) ** 2))

        # sample using the reparameterization trick
        actions = self._b_distribution.rsample()

        # If the actions are coming from the buffer, we need to rescale them to be in the range [0, 1]
        taken_actions = inputs.get("taken_actions", None)
        if taken_actions is not None:
            taken_actions = (taken_actions - self._b_actions_min) / (self._b_actions_max - self._b_actions_min)
        else:
            taken_actions = actions

        # clip actions to be in the range ]0, 1[
        taken_actions = taken_actions.clamp(min=EPS, max=1 - EPS)
        # log of the probability density function
        log_prob = self._b_distribution.log_prob(taken_actions)

        if self._b_reduction is not None:
            log_prob = self._b_reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)
        outputs["mean_actions"] = (a / (a + b)) * (self._b_actions_max - self._b_actions_min) + self._b_actions_min
        actions = actions * (self._b_actions_max - self._b_actions_min) + self._b_actions_min
        return actions, log_prob, outputs

    def get_entropy(self, role: str = "") -> torch.Tensor:
        """Compute and return the entropy of the model

        :return: Entropy of the model
        :rtype: torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> entropy = model.get_entropy()
            >>> print(entropy.shape)
            torch.Size([4096, 8])
        """
        if self._b_distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._b_distribution.entropy().to(self.device)

    def get_log_std(self, role: str = "") -> torch.Tensor:
        """Return the log standard deviation of the model

        :return: Log standard deviation of the model
        :rtype: torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> log_std = model.get_log_std()
            >>> print(log_std.shape)
            torch.Size([4096, 8])
        """
        return self._b_log_std

    def distribution(self, role: str = "") -> torch.distributions.Beta:
        """Get the current distribution of the model

        :return: Distribution of the model
        :rtype: torch.distributions.Beta
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> distribution = model.distribution()
            >>> print(distribution)
            Beta(alpha: torch.Size([4096, 8]), beta: torch.Size([4096, 8]))
        """
        return self._b_distribution
