from typing import Any, Mapping, Tuple, Union

import gymnasium

import torch


class DeterministicMixin:
    def __init__(self, *, clip_actions: bool = False, role: str = "") -> None:
        """Deterministic mixin model (deterministic model).

        :param clip_actions: Flag to indicate whether the actions should be clipped to the action space.
        :param role: Role played by the model.
        """
        self._clip_actions = clip_actions and isinstance(self.action_space, gymnasium.Space)

        if self._clip_actions:
            self._clip_actions_min = torch.tensor(self.action_space.low, device=self.device, dtype=torch.float32)
            self._clip_actions_max = torch.tensor(self.action_space.high, device=self.device, dtype=torch.float32)

    def act(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act deterministically in response to the observations/states of the environment.

        :param inputs: Model inputs. The most common keys are:

            - ``"observations"``: observation of the environment used to make the decision.
            - ``"states"``: state of the environment used to make the decision.
            - ``"taken_actions"``: actions taken by the policy for the given observations/states.
        :param role: Role played by the model.

        :return: Model output. The first component is the expected action/value returned by the model.
            The second component is a dictionary containing extra output values according to the model.
        """
        # map from observations/states to actions
        actions, outputs = self.compute(inputs, role)

        # clip actions
        if self._clip_actions:
            actions = torch.clamp(actions, min=self._clip_actions_min, max=self._clip_actions_max)

        return actions, outputs
