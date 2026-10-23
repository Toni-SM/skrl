from __future__ import annotations

from typing import Callable

import dataclasses

from skrl.multi_agents.torch import MultiAgentCfg


@dataclasses.dataclass(kw_only=True)
class HAD3QN_CFG(MultiAgentCfg):
    """Configuration for the HAD3QN agent."""

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

    It applies to the target networks of both the Q-network and the critic. Since the critic is shared by all
    the agents, it must have the same value for all the agents.
    """

    learning_rate: float | tuple[float, float] | dict[str, float | tuple[float, float]] = (5e-4, 1e-3)
    """Learning rate for the Q-network and critic networks.

    * If a float is provided, the same learning rate will be used for the networks.
    * If a tuple is provided, its elements will be used for each network in order.
    """

    learning_rate_scheduler: (
        type | tuple[type | None, type | None] | dict[str, type | tuple[type | None, type | None]] | None
    ) = None
    """Learning rate scheduler class for the Q-network and critic networks.

    See :ref:`learning_rate_schedulers` for more details.

    * If a class is provided, the same learning rate scheduler will be used for the networks.
    * If a tuple is provided, its elements will be used for each network in order.
    """

    learning_rate_scheduler_kwargs: dict | tuple[dict, dict] | dict[str, dict | tuple[dict, dict]] = dataclasses.field(
        default_factory=dict
    )
    """Keyword arguments for the learning rate scheduler's constructor.

    See :ref:`learning_rate_schedulers` for more details.

    .. warning::

        The ``optimizer`` argument is automatically passed to the learning rate scheduler's constructor.
        Therefore, it must not be provided in the keyword arguments.

    * If a dictionary is provided, the same keyword arguments will be used for the networks.
    * If a tuple is provided, its elements will be used for each network in order.
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

    update_interval: int = 1
    """Number of environment steps to perform between algorithm updates."""

    target_update_interval: int = 10
    """Number of algorithm updates to perform between target network updates."""

    exploration_scheduler: Callable[[int, int], float] | None | dict[str, Callable[[int, int], float] | None] = None
    """Epsilon-greedy exploration scheduler function.

    The function takes the current ``timestep`` and the total number of ``timesteps`` as arguments
    and returns an epsilon value used to sample greedy actions.
    """

    rewards_shaper: Callable | None = None
    """Rewards shaping function."""

    mixed_precision: bool = False
    """Whether to enable automatic mixed precision for higher performance."""

    fixed_order: bool = False
    """Whether to update the agents' Q-networks in a fixed order.

    * If True, the Q-networks are updated sequentially in the order of the possible agents.
    * If False, the Q-networks are updated sequentially in a new random order at each gradient step.
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
                "update_interval",
                "target_update_interval",
            ],
        )
        for uid in possible_agents:
            # learning rate
            if not isinstance(self.learning_rate[uid], (tuple, list)):
                self.learning_rate[uid] = (self.learning_rate[uid], self.learning_rate[uid])
            # learning rate scheduler
            if self.learning_rate_scheduler[uid] is None:
                self.learning_rate_scheduler[uid] = (None, None)
            elif not isinstance(self.learning_rate_scheduler[uid], (tuple, list)):
                self.learning_rate_scheduler[uid] = (
                    self.learning_rate_scheduler[uid],
                    self.learning_rate_scheduler[uid],
                )
            # learning rate scheduler kwargs
            if not isinstance(self.learning_rate_scheduler_kwargs[uid], (tuple, list)):
                self.learning_rate_scheduler_kwargs[uid] = (
                    self.learning_rate_scheduler_kwargs[uid],
                    self.learning_rate_scheduler_kwargs[uid],
                )
