from typing import Optional, Union, Sequence

import gym

import torch
from torch.distributions import Categorical

from . import Model


class CategoricalModel(Model):
    def __init__(self, 
                 observation_space: Union[int, Sequence[int], gym.Space], 
                 action_space: Union[int, Sequence[int], gym.Space], 
                 device: Union[str, torch.device] = "cuda:0",
                 unnormalized_log_prob: bool = True) -> None:
        """Categorical model (stochastic model)

        :param observation_space: Observation/state space or shape.
                                  The ``num_observations`` property will contain the size of that space
        :type observation_space: int, sequence of int, gym.Space
        :param action_space: Action space or shape.
                             The ``num_actions`` property will contain the size of that space
        :type action_space: int, sequence of int, gym.Space
        :param device: Device on which a torch tensor is or will be allocated (default: ``"cuda:0"``)
        :type device: str or torch.device, optional
        :param unnormalized_log_prob: Flag to indicate how to be interpreted the model's output (default: ``True``).
                                      If True, the model's output is interpreted as unnormalized log probabilities 
                                      (it can be any real number), otherwise as normalized probabilities 
                                      (the output must be non-negative, finite and have a non-zero sum)
        :type unnormalized_log_prob: bool, optional

        Example::

            # define the model
            >>> import torch
            >>> import torch.nn as nn
            >>> from skrl.models.torch import CategoricalModel
            >>> 
            >>> class Policy(CategoricalModel):
            ...     def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
            ...         super().__init__(observation_space, action_space, device, unnormalized_log_prob)
            ...
            ...         self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
            ...                                  nn.ELU(),
            ...                                  nn.Linear(32, 32),
            ...                                  nn.ELU(),
            ...                                  nn.Linear(32, self.num_actions))
            ...
            ...     def compute(self, states, taken_actions, role):
            ...         return self.net(states)
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
        super(CategoricalModel, self).__init__(observation_space, action_space, device)

        self._unnormalized_log_prob = unnormalized_log_prob

        self._distribution = None

    def act(self, 
            states: torch.Tensor, 
            taken_actions: Optional[torch.Tensor] = None, 
            inference: bool = False,
            role: str = "") -> Sequence[torch.Tensor]:
        """Act stochastically in response to the state of the environment

        :param states: Observation/state of the environment used to make the decision
        :type states: torch.Tensor
        :param taken_actions: Actions taken by a policy to the given states (default: ``None``).
                              The use of these actions only makes sense in critical models, e.g.
        :type taken_actions: torch.Tensor, optional
        :param inference: Flag to indicate whether the model is making inference (default: ``False``)
        :type inference: bool, optional
        :param role: Role of the model (default: ``""``)
        :type role: str, optional

        :return: Action to be taken by the agent given the state of the environment.
                 The sequence's components are the actions, the log of the probability density function and the model's output
        :rtype: sequence of torch.Tensor

        Example::

            >>> # given a batch of sample states with shape (4096, 4)
            >>> action, log_prob, net_output = model.act(states)
            >>> print(action.shape, log_prob.shape, net_output.shape)
            torch.Size([4096, 1]) torch.Size([4096, 1]) torch.Size([4096, 2])
        """
        # map from states/observations to normalized probabilities or unnormalized log probabilities
        if self._instantiator_net is None:
            output = self.compute(states.to(self.device), 
                                  taken_actions.to(self.device) if taken_actions is not None else taken_actions,
                                  role)
        else:
            output = self._get_instantiator_output(states.to(self.device), \
                taken_actions.to(self.device) if taken_actions is not None else taken_actions)

        # unnormalized log probabilities
        if self._unnormalized_log_prob:
            self._distribution = Categorical(logits=output)
        # normalized probabilities
        else:
            self._distribution = Categorical(probs=output)
        
        # actions and log of the probability density function
        actions = self._distribution.sample()
        log_prob = self._distribution.log_prob(actions if taken_actions is None else taken_actions.view(-1))

        if inference:
            return actions.unsqueeze(-1).detach(), log_prob.unsqueeze(-1).detach(), output.detach()
        return actions.unsqueeze(-1), log_prob.unsqueeze(-1), output

    def distribution(self) -> torch.distributions.Categorical:
        """Get the current distribution of the model

        :return: Distribution of the model
        :rtype: torch.distributions.Categorical

        Example::

            >>> distribution = model.distribution()
            >>> print(distribution)
            Categorical(probs: torch.Size([4096, 2]), logits: torch.Size([4096, 2]))
        """
        return self._distribution