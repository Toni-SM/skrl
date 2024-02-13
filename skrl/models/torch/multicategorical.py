from typing import Any, Mapping, Sequence, Tuple, Union

import torch
from torch.distributions import Categorical


class MultiCategoricalMixin:
    def __init__(self, unnormalized_log_prob: bool = True, reduction: str = "sum", role: str = "") -> None:
        """MultiCategorical mixin model (stochastic model)

        :param unnormalized_log_prob: Flag to indicate how to be interpreted the model's output (default: ``True``).
                                      If True, the model's output is interpreted as unnormalized log probabilities
                                      (it can be any real number), otherwise as normalized probabilities
                                      (the output must be non-negative, finite and have a non-zero sum)
        :type unnormalized_log_prob: bool, optional
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
            >>> from skrl.models.torch import Model, MultiCategoricalMixin
            >>>
            >>> class Policy(MultiCategoricalMixin, Model):
            ...     def __init__(self, observation_space, action_space, device="cuda:0", unnormalized_log_prob=True, reduction="sum"):
            ...         Model.__init__(self, observation_space, action_space, device)
            ...         MultiCategoricalMixin.__init__(self, unnormalized_log_prob, reduction)
            ...
            ...         self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
            ...                                  nn.ELU(),
            ...                                  nn.Linear(32, 32),
            ...                                  nn.ELU(),
            ...                                  nn.Linear(32, self.num_actions))
            ...
            ...     def compute(self, inputs, role):
            ...         return self.net(inputs["states"]), {}
            ...
            >>> # given an observation_space: gym.spaces.Box with shape (4,)
            >>> # and an action_space: gym.spaces.MultiDiscrete with nvec = [3, 2]
            >>> model = Policy(observation_space, action_space)
            >>>
            >>> print(model)
            Policy(
              (net): Sequential(
                (0): Linear(in_features=4, out_features=32, bias=True)
                (1): ELU(alpha=1.0)
                (2): Linear(in_features=32, out_features=32, bias=True)
                (3): ELU(alpha=1.0)
                (4): Linear(in_features=32, out_features=5, bias=True)
              )
            )
        """
        self._unnormalized_log_prob = unnormalized_log_prob
        self._distributions = []

        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._reduction = torch.mean if reduction == "mean" else torch.sum if reduction == "sum" \
            else torch.prod if reduction == "prod" else None

    def act(self,
            inputs: Mapping[str, Union[torch.Tensor, Any]],
            role: str = "") -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act stochastically in response to the state of the environment

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function.
                 The third component is a dictionary containing the network output ``"net_output"``
                 and extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dict

        Example::

            >>> # given a batch of sample states with shape (4096, 4)
            >>> actions, log_prob, outputs = model.act({"states": states})
            >>> print(actions.shape, log_prob.shape, outputs["net_output"].shape)
            torch.Size([4096, 2]) torch.Size([4096, 1]) torch.Size([4096, 5])
        """
        # map from states/observations to normalized probabilities or unnormalized log probabilities
        net_output, outputs = self.compute(inputs, role)

        # unnormalized log probabilities
        if self._unnormalized_log_prob:
            self._distributions = [Categorical(logits=logits) for logits in torch.split(net_output, self.action_space.nvec.tolist(), dim=-1)]
        # normalized probabilities
        else:
            self._distributions = [Categorical(probs=probs) for probs in torch.split(net_output, self.action_space.nvec.tolist(), dim=-1)]

        # actions
        actions = torch.stack([distribution.sample() for distribution in self._distributions], dim=-1)

        # log of the probability density function
        log_prob = torch.stack([distribution.log_prob(_actions.view(-1)) for _actions, distribution \
                                in zip(torch.unbind(inputs.get("taken_actions", actions), dim=-1), self._distributions)], dim=-1)
        if self._reduction is not None:
            log_prob = self._reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["net_output"] = net_output
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
            torch.Size([4096, 1])
        """
        if self._distributions:
            entropy = torch.stack([distribution.entropy().to(self.device) for distribution in self._distributions], dim=-1)
            if self._reduction is not None:
                return self._reduction(entropy, dim=-1).unsqueeze(-1)
            return entropy
        return torch.tensor(0.0, device=self.device)

    def distribution(self, role: str = "") -> torch.distributions.Categorical:
        """Get the current distribution of the model

        :return: First distributions of the model
        :rtype: torch.distributions.Categorical
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> distribution = model.distribution()
            >>> print(distribution)
            Categorical(probs: torch.Size([10, 3]), logits: torch.Size([10, 3]))
        """
        # TODO: find a way to integrate in the class the distribution functions (e.g.: stddev)
        return self._distributions[0]
