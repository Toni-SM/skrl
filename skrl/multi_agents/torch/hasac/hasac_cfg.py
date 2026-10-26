from __future__ import annotations

from typing import Callable

import dataclasses

from skrl.multi_agents.torch import MultiAgentCfg


@dataclasses.dataclass(kw_only=True)
class HASAC_CFG(MultiAgentCfg):
    """Configuration for the HASAC agent."""

    gradient_steps: int = 1
    """Number of gradient steps to perform for each update."""

    batch_size: int = 64
    """Batch size for sampling transitions from memory during training."""

    discount_factor: float | dict[str, float] = 0.99
    """Parameter that balances the importance of future rewards (close to 1.0) versus immediate rewards (close to 0.0).

    Range: ``[0.0, 1.0]``.
    """

    polyak: float | dict[str, float] = 0.005
    """Parameter to control the update of the target networks by polyak averaging.

    Range: ``[0.0, 1.0]``. See :py:meth:`~skrl.models.torch.base.Model.update_parameters` for more details.

    It applies to the target critics only (there is no target policy). When the critic is shared by all the agents,
    it must have the same value for all the agents.
    """

    learning_rate: float | tuple[float, float, float] | dict[str, float | tuple[float, float, float]] = 1e-3
    """Learning rate for the actor and critic networks, and entropy coefficient.

    * If a float is provided, the same learning rate will be used for the networks/coefficient.
    * If a tuple is provided, its elements will be used for each network/coefficient in order.
    """

    learning_rate_scheduler: (
        type
        | tuple[type | None, type | None, type | None]
        | dict[str, type | tuple[type | None, type | None, type | None]]
        | None
    ) = None
    """Learning rate scheduler class for the actor and critic networks, and entropy coefficient.

    See :ref:`learning_rate_schedulers` for more details.

    * If a class is provided, the same learning rate scheduler will be used for the networks/coefficient.
    * If a tuple is provided, its elements will be used for each network/coefficient in order.
    """

    learning_rate_scheduler_kwargs: dict | tuple[dict, dict, dict] | dict[str, dict | tuple[dict, dict, dict]] = (
        dataclasses.field(default_factory=dict)
    )
    """Keyword arguments for the learning rate scheduler's constructor.

    See :ref:`learning_rate_schedulers` for more details.

    .. warning::

        The ``optimizer`` argument is automatically passed to the learning rate scheduler's constructor.
        Therefore, it must not be provided in the keyword arguments.

    * If a dictionary is provided, the same keyword arguments will be used for the networks/coefficient.
    * If a tuple is provided, its elements will be used for each network/coefficient in order.
    """

    observation_preprocessor: type | None | dict[str, type | None] = None
    """Preprocessor class to process the environment's observations.

    See :ref:`preprocessors` for more details.
    """

    observation_preprocessor_kwargs: dict | dict[str, dict] = dataclasses.field(default_factory=dict)
    """Keyword arguments for the observation preprocessor's constructor.

    See :ref:`preprocessors` for more details.
    """

    state_preprocessor: type | None | dict[str, type | None] = None
    """Preprocessor class to process the environment's states.

    See :ref:`preprocessors` for more details.
    """

    state_preprocessor_kwargs: dict | dict[str, dict] = dataclasses.field(default_factory=dict)
    """Keyword arguments for the state preprocessor's constructor.

    See :ref:`preprocessors` for more details.
    """

    random_timesteps: int = 0
    """Number of random exploration (sampling random actions) steps to perform before sampling actions from the policy."""

    learning_starts: int = 0
    """Number of steps to perform before calling the algorithm update function."""

    grad_norm_clip: float | dict[str, float] = 0
    """Clipping coefficient for the gradients by their global norm.

    If less than or equal to 0, the gradients will not be clipped.

    It applies to the gradients of both the policy and the critic. When the critic is shared by all the agents,
    it must have the same value for all the agents.
    """

    learn_entropy: bool | dict[str, bool] = True
    """Whether to learn the entropy coefficient."""

    initial_entropy_value: float | dict[str, float] = 0.2
    """Initial value for the entropy coefficient."""

    target_entropy: float | None | dict[str, float | None] = None
    """Target value for computing the entropy loss."""

    rewards_shaper: Callable | None = None
    """Rewards shaping function."""

    mixed_precision: bool = False
    """Whether to enable automatic mixed precision for higher performance."""

    fixed_order: bool = False
    """Whether to update the agents' policies in a fixed order.

    * If True, the policies are updated sequentially in the order of the possible agents.
    * If False, the policies are updated sequentially in a new random order at each gradient step.
    """

    def expand(self, *, possible_agents: list[str]) -> None:
        """Expand the configuration."""
        super().expand(
            possible_agents=possible_agents,
            immutable=[
                "mixed_precision",
                "rewards_shaper",
                "fixed_order",
                "batch_size",
                "gradient_steps",
                "random_timesteps",
                "learning_starts",
            ],
        )
        for uid in possible_agents:
            # learning rate
            if not isinstance(self.learning_rate[uid], (tuple, list)):
                self.learning_rate[uid] = (self.learning_rate[uid], self.learning_rate[uid], self.learning_rate[uid])
            # learning rate scheduler
            if self.learning_rate_scheduler[uid] is None:
                self.learning_rate_scheduler[uid] = (None, None, None)
            elif not isinstance(self.learning_rate_scheduler[uid], (tuple, list)):
                self.learning_rate_scheduler[uid] = (
                    self.learning_rate_scheduler[uid],
                    self.learning_rate_scheduler[uid],
                    self.learning_rate_scheduler[uid],
                )
            # learning rate scheduler kwargs
            if not isinstance(self.learning_rate_scheduler_kwargs[uid], (tuple, list)):
                self.learning_rate_scheduler_kwargs[uid] = (
                    self.learning_rate_scheduler_kwargs[uid],
                    self.learning_rate_scheduler_kwargs[uid],
                    self.learning_rate_scheduler_kwargs[uid],
                )
