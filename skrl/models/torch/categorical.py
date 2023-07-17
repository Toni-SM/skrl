from typing import Any, Mapping, Tuple, Union

import torch
from torch.distributions import Categorical


class CategoricalMixin:
    def __init__(self, unnormalized_log_prob: bool = True, role: str = "") -> None:
        """Categorical mixin model (stochastic model)

        :param unnormalized_log_prob: Flag to indicate how to be interpreted the model's output (default: ``True``).
                                      If True, the model's output is interpreted as unnormalized log probabilities
                                      (it can be any real number), otherwise as normalized probabilities
                                      (the output must be non-negative, finite and have a non-zero sum)
        :type unnormalized_log_prob: bool, optional
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            # define the model
            >>> import torch
            >>> import torch.nn as nn
            >>> from skrl.models.torch import Model, CategoricalMixin
            >>>
            >>> class Policy(CategoricalMixin, Model):
            ...     def __init__(self, observation_space, action_space, device="cuda:0", unnormalized_log_prob=True):
            ...         Model.__init__(self, observation_space, action_space, device)
            ...         CategoricalMixin.__init__(self, unnormalized_log_prob)
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
            >>> # and an action_space: gym.spaces.Discrete with n = 2
            >>> model = Policy(observation_space, action_space)
            >>>
            >>> print(model)
            Policy(
              (net): Sequential(
                (0): Linear(in_features=4, out_features=32, bias=True)
                (1): ELU(alpha=1.0)
                (2): Linear(in_features=32, out_features=32, bias=True)
                (3): ELU(alpha=1.0)
                (4): Linear(in_features=32, out_features=2, bias=True)
              )
            )
        """
        if not hasattr(self, "_c_unnormalized_log_prob"):
            self._c_unnormalized_log_prob = {}
        self._c_unnormalized_log_prob[role] = unnormalized_log_prob

        if not hasattr(self, "_c_distribution"):
            self._c_distribution = {}
        self._c_distribution[role] = None

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
            torch.Size([4096, 1]) torch.Size([4096, 1]) torch.Size([4096, 2])
        """
        # map from states/observations to normalized probabilities or unnormalized log probabilities
        net_output, outputs = self.compute(inputs, role)

        # unnormalized log probabilities
        if self._c_unnormalized_log_prob[role] if role in self._c_unnormalized_log_prob else self._c_unnormalized_log_prob[""]:
            self._c_distribution[role] = Categorical(logits=net_output)
        # normalized probabilities
        else:
            self._c_distribution[role] = Categorical(probs=net_output)

        # actions and log of the probability density function
        actions = self._c_distribution[role].sample()
        log_prob = self._c_distribution[role].log_prob(inputs.get("taken_actions", actions).view(-1))

        outputs["net_output"] = net_output
        return actions.unsqueeze(-1), log_prob.unsqueeze(-1), outputs

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
        distribution = self._c_distribution[role] if role in self._c_distribution else self._c_distribution[""]
        if distribution is None:
            return torch.tensor(0.0, device=self.device)
        return distribution.entropy().to(self.device)

    def distribution(self, role: str = "") -> torch.distributions.Categorical:
        """Get the current distribution of the model

        :return: Distribution of the model
        :rtype: torch.distributions.Categorical
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> distribution = model.distribution()
            >>> print(distribution)
            Categorical(probs: torch.Size([4096, 2]), logits: torch.Size([4096, 2]))
        """
        return self._c_distribution[role] if role in self._c_distribution else self._c_distribution[""]
