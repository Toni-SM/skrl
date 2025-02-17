from typing import Any, Mapping, Tuple, Union

import torch
from torch.distributions import Categorical


# speed up distribution construction by disabling checking
Categorical.set_default_validate_args(False)


class CategoricalMixin:
    def __init__(self, *, unnormalized_log_prob: bool = True, role: str = "") -> None:
        """Categorical mixin model (stochastic model).

        :param unnormalized_log_prob: Flag to indicate how to the model's output will be interpreted.
            If True, the model's output is interpreted as unnormalized log probabilities (it can be any real number),
            otherwise as normalized probabilities (the output must be non-negative, finite and have a non-zero sum).
        :param role: Role played by the model.
        """
        self._unnormalized_log_prob = unnormalized_log_prob
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

            - ``"log_prob"``: log of the probability density function.
            - ``"net_output"``: network output.
        """
        # map from observations/states to normalized probabilities or unnormalized log probabilities
        net_output, outputs = self.compute(inputs, role)

        # unnormalized log probabilities
        if self._unnormalized_log_prob:
            self._distribution = Categorical(logits=net_output)
        # normalized probabilities
        else:
            self._distribution = Categorical(probs=net_output)

        # actions and log of the probability density function
        actions = self._distribution.sample()
        log_prob = self._distribution.log_prob(inputs.get("taken_actions", actions).view(-1))

        outputs["log_prob"] = log_prob.unsqueeze(-1)
        outputs["net_output"] = net_output
        return actions.unsqueeze(-1), outputs

    def get_entropy(self, *, role: str = "") -> torch.Tensor:
        """Compute and return the entropy of the model.

        :param role: Role played by the model.

        :return: Entropy of the model.
        """
        if self._distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._distribution.entropy().to(self.device)

    def distribution(self, *, role: str = "") -> torch.distributions.Categorical:
        """Get the current distribution of the model.

        :param role: Role played by the model.

        :return: Distribution of the model.
        """
        return self._distribution
