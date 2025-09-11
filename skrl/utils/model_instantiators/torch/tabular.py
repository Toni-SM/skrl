from typing import Any, Literal, Mapping, Optional, Union

import gymnasium

import torch
import torch.nn as nn  # noqa

from skrl.models.torch import TabularMixin  # noqa
from skrl.models.torch import Model


def tabular_model(
    *,
    observation_space: Optional[gymnasium.Space] = None,
    state_space: Optional[gymnasium.Space] = None,
    action_space: Optional[gymnasium.Space] = None,
    device: Optional[Union[str, torch.device]] = None,
    variant: Literal["epsilon-greedy"] = "epsilon-greedy",
    variant_kwargs: Mapping[str, Any] = {},
    return_source: bool = False,
) -> Union[Model, str]:
    """Instantiate a :class:`~skrl.models.torch.tabular.TabularMixin`-based model.

    Supported variants:

    - ``epsilon-greedy``: Simple method of balancing exploration and exploitation by randomly selecting one or the other.

      .. list-table::
          :header-rows: 1

          * - Argument
            - Type
            - Default
            - Description
          * - ``epsilon``
            - ``float``
            - ``0.1``
            - Cut-off probability for choosing to explore

    :param observation_space: Observation space. The ``num_observations`` property will contain the size of the space.
    :param state_space: State space. The ``num_states`` property will contain the size of the space.
    :param action_space: Action space. The ``num_actions`` property will contain the size of the space.
    :param device: Data allocation and computation device. If not specified, the default device will be used.
    :param variant: Variant of the model.
    :param variant_kwargs: Variant-specific keyword arguments.
    :param return_source: Whether to return the source string containing the model class used to
        instantiate the model rather than the model instance.

    :return: Tabular model instance or definition source (if ``return_source`` is True).
    """

    if variant == "epsilon-greedy":
        epsilon = float(variant_kwargs.get("epsilon", 0.1))
        template = f"""class TabularModel(TabularMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, role=""):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        TabularMixin.__init__(self, role=role)

        self.q_table = nn.Parameter(
            torch.ones((self.num_observations, self.num_actions), dtype=torch.float32, device=self.device),
            requires_grad=False,
        )

    def compute(self, inputs, role=""):
        actions = torch.argmax(self.q_table[inputs["observations"]], dim=-1, keepdim=False)

        # choose random actions for exploration according to epsilon
        indexes = (torch.rand(inputs["observations"].shape[0], device=self.device) < {epsilon}).nonzero().flatten()
        if indexes.numel():
            actions[indexes] = torch.randint(self.num_actions, (indexes.numel(), 1), device=self.device)
        return actions, {{}}
        """
    else:
        raise ValueError(f"Invalid variant: '{variant}'")

    # return source
    if return_source:
        return template

    # instantiate model
    _locals = {}
    exec(template, globals(), _locals)
    return _locals["TabularModel"](
        observation_space=observation_space, state_space=state_space, action_space=action_space, device=device
    )
