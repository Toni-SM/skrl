from typing import Any, Mapping, Tuple, Union

import torch
from torch.distributions import Categorical


# speed up distribution construction by disabling checking
Categorical.set_default_validate_args(False)


class MultiCategoricalMixin:
    def __init__(self, *, unnormalized_log_prob: bool = True, reduction: str = "sum", role: str = "") -> None:
        """MultiCategorical mixin model (stochastic model).

        :param unnormalized_log_prob: Flag to indicate how to the model's output will be interpreted.
            If True, the model's output is interpreted as unnormalized log probabilities (it can be any real number),
            otherwise as normalized probabilities (the output must be non-negative, finite and have a non-zero sum).
        :param reduction: Reduction method for returning the log probability density function.
            Supported values are ``"mean"``, ``"sum"``, ``"prod"`` and ``"none"``.
            If ``"none"``, the log probability density function is returned as a tensor of shape
            ``(num_samples, num_actions)`` instead of ``(num_samples, 1)``.
        :param role: Role played by the model.

        :raises ValueError: If the reduction method is not valid
        """
        self._mc_unnormalized_log_prob = unnormalized_log_prob
        self._mc_distributions = []

        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("Reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._mc_reduction = (
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

            - ``"log_prob"``: log of the probability density function.
            - ``"net_output"``: network output.
        """
        # map from observations/states to normalized probabilities or unnormalized log probabilities
        net_output, outputs = self.compute(inputs, role)

        # unnormalized log probabilities
        if self._mc_unnormalized_log_prob:
            self._mc_distributions = [
                Categorical(logits=logits)
                for logits in torch.split(net_output, self.action_space.nvec.tolist(), dim=-1)
            ]
        # normalized probabilities
        else:
            self._mc_distributions = [
                Categorical(probs=probs) for probs in torch.split(net_output, self.action_space.nvec.tolist(), dim=-1)
            ]

        # actions
        actions = torch.stack([distribution.sample() for distribution in self._mc_distributions], dim=-1)

        # log of the probability density function
        log_prob = torch.stack(
            [
                distribution.log_prob(_actions.view(-1))
                for _actions, distribution in zip(
                    torch.unbind(inputs.get("taken_actions", actions), dim=-1), self._mc_distributions
                )
            ],
            dim=-1,
        )
        if self._mc_reduction is not None:
            log_prob = self._mc_reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["log_prob"] = log_prob
        outputs["net_output"] = net_output
        return actions, outputs

    def get_entropy(self, *, role: str = "") -> torch.Tensor:
        """Compute and return the entropy of the model.

        :param role: Role played by the model.

        :return: Entropy of the model.
        """
        if self._mc_distributions:
            entropy = torch.stack(
                [distribution.entropy().to(self.device) for distribution in self._mc_distributions], dim=-1
            )
            if self._mc_reduction is not None:
                return self._mc_reduction(entropy, dim=-1).unsqueeze(-1)
            return entropy
        return torch.tensor(0.0, device=self.device)

    def distribution(self, *, role: str = "") -> torch.distributions.Categorical:
        """Get the current distribution of the model.

        :param role: Role played by the model.

        :return: First distribution of the model.
        """
        # TODO: find a way to return all distributions
        # TODO: find a way to integrate in the class the distribution functions (e.g.: stddev)
        return self._mc_distributions[0]
