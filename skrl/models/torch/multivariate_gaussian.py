from typing import Any, Mapping, Tuple, Union

import gymnasium

import torch
from torch.distributions import MultivariateNormal


# speed up distribution construction by disabling checking
MultivariateNormal.set_default_validate_args(False)


class MultivariateGaussianMixin:
    def __init__(
        self,
        *,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        role: str = "",
    ) -> None:
        """Multivariate Gaussian mixin model (stochastic model).

        :param clip_actions: Flag to indicate whether the actions should be clipped to the action space.
        :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped.
        :param min_log_std: Minimum value of the log standard deviation if ``clip_log_std`` is True.
        :param max_log_std: Maximum value of the log standard deviation if ``clip_log_std`` is True.
        :param role: Role played by the model.
        """
        self._clip_actions = clip_actions and isinstance(self.action_space, gymnasium.Space)

        if self._clip_actions:
            self._clip_actions_min = torch.tensor(self.action_space.low, device=self.device, dtype=torch.float32)
            self._clip_actions_max = torch.tensor(self.action_space.high, device=self.device, dtype=torch.float32)

        self._clip_log_std = clip_log_std
        self._log_std_min = min_log_std
        self._log_std_max = max_log_std

        self._distribution = None

    def act(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], *, role: str = ""
    ) -> Tuple[torch.Tensor, Mapping[str, Union[torch.Tensor, Any]]]:
        """Act stochastically in response to the observations/states of the environment.

        :param inputs: Model inputs. The most common keys are:

            - ``"observations"``: observation of the environment used to make the decision.
            - ``"states"``: state of the environment used to make the decision.
            - ``"taken_actions"``: actions taken by the policy for the given observations/states.
        :param role: Role played by the model.

        :return: Model output. The first component is the expected action/value returned by the model.
            The second component is a dictionary containing the following extra output values:

            - ``"log_std"``: log of the standard deviation.
            - ``"log_prob"``: log of the probability density function.
            - ``"mean_actions"``: mean actions (network output).
        """
        # map from observations/states to mean actions and log standard deviations
        mean_actions, outputs = self.compute(inputs, role)
        log_std = outputs["log_std"]

        # clamp log standard deviations
        if self._clip_log_std:
            log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)
            outputs["log_std"] = log_std

        # distribution
        covariance = torch.diag(log_std.exp() * log_std.exp())
        self._distribution = MultivariateNormal(mean_actions, scale_tril=covariance)

        # sample using the reparameterization trick
        actions = self._distribution.rsample()

        # clip actions
        if self._clip_actions:
            actions = torch.clamp(actions, min=self._clip_actions_min, max=self._clip_actions_max)

        # log of the probability density function
        log_prob = self._distribution.log_prob(inputs.get("taken_actions", actions))
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["log_prob"] = log_prob
        outputs["mean_actions"] = mean_actions
        return actions, log_prob, outputs

    def get_entropy(self, *, role: str = "") -> torch.Tensor:
        """Compute and return the entropy of the model.

        :param role: Role played by the model.

        :return: Entropy of the model.
        """
        if self._distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._distribution.entropy().to(self.device)

    def distribution(self, *, role: str = "") -> torch.distributions.MultivariateNormal:
        """Get the current distribution of the model.

        :param role: Role played by the model.

        :return: Distribution of the model.
        """
        return self._distribution
