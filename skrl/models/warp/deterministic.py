from __future__ import annotations

from typing import Any

import warp as wp

from skrl.utils.framework.warp import clamp
from skrl.utils.spaces.warp import compute_space_limits


class DeterministicMixin:
    def __init__(self, *, clip_actions: bool = False, role: str = "") -> None:
        """Deterministic mixin model (deterministic model).

        :param clip_actions: Flag to indicate whether the actions should be clipped to the action space.
        :param role: Role played by the model.
        """
        self._d_clip_actions = clip_actions
        self._d_clip_actions_min, self._d_clip_actions_max = compute_space_limits(self.action_space, device=self.device)

    def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[wp.array, dict[str, Any]]:
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
        if self._d_clip_actions:
            actions = clamp(actions, min=self._d_clip_actions_min, max=self._d_clip_actions_max)

        return actions, outputs
