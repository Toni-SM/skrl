from typing import Any, Mapping, Tuple, Union

import gymnasium

import torch
from torch.distributions import Normal


# speed up distribution construction by disabling checking
Normal.set_default_validate_args(False)


class GaussianMixin:
    def __init__(
        self,
        *,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: str = "sum",
        role: str = "",
    ) -> None:
        """Gaussian mixin model (stochastic model).

        :param clip_actions: Flag to indicate whether the actions should be clipped to the action space.
        :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped.
        :param min_log_std: Minimum value of the log standard deviation if ``clip_log_std`` is True.
        :param max_log_std: Maximum value of the log standard deviation if ``clip_log_std`` is True.
        :param reduction: Reduction method for returning the log probability density function.
            Supported values are ``"mean"``, ``"sum"``, ``"prod"`` and ``"none"``.
            If ``"none"``, the log probability density function is returned as a tensor of shape
            ``(num_samples, num_actions)`` instead of ``(num_samples, 1)``.
        :param role: Role played by the model.

        :raises ValueError: If the reduction method is not valid.
        """
        self._g_clip_actions = clip_actions and isinstance(self.action_space, gymnasium.Space)

        if self._g_clip_actions:
            self._g_clip_actions_min = torch.tensor(self.action_space.low, device=self.device, dtype=torch.float32)
            self._g_clip_actions_max = torch.tensor(self.action_space.high, device=self.device, dtype=torch.float32)

        self._g_clip_log_std = clip_log_std
        self._g_log_std_min = min_log_std
        self._g_log_std_max = max_log_std

        self._g_distribution = None

        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("Reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._g_reduction = (
            torch.mean
            if reduction == "mean"
            else torch.sum if reduction == "sum" else torch.prod if reduction == "prod" else None
        )

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
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_log_std_min, self._g_log_std_max)
            outputs["log_std"] = log_std

        # distribution
        self._g_distribution = Normal(mean_actions, log_std.exp())

        # sample using the reparameterization trick
        actions = self._g_distribution.rsample()

        # clip actions
        if self._g_clip_actions:
            actions = torch.clamp(actions, min=self._g_clip_actions_min, max=self._g_clip_actions_max)

        # log of the probability density function
        log_prob = self._g_distribution.log_prob(inputs.get("taken_actions", actions))
        if self._g_reduction is not None:
            log_prob = self._g_reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["log_prob"] = log_prob
        outputs["mean_actions"] = mean_actions
        return actions, outputs

    def get_entropy(self, *, role: str = "") -> torch.Tensor:
        """Compute and return the entropy of the model.

        :param role: Role played by the model.

        :return: Entropy of the model.
        """
        if self._g_distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._g_distribution.entropy().to(self.device)

    def distribution(self, *, role: str = "") -> torch.distributions.Normal:
        """Get the current distribution of the model.

        :param role: Role played by the model.

        :return: Distribution of the model.
        """
        return self._g_distribution
