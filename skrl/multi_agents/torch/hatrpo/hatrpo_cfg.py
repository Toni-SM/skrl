from __future__ import annotations

from typing import Callable

import dataclasses

from skrl.multi_agents.torch import MultiAgentCfg


@dataclasses.dataclass(kw_only=True)
class HATRPO_CFG(MultiAgentCfg):
    """Configuration for the HATRPO agent."""

    rollouts: int = 16
    """Number of collection steps to perform between updates."""

    learning_epochs: int | dict[str, int] = 8
    """Number of learning epochs to perform during updates."""

    mini_batches: int | dict[str, int] = 2
    """Number of mini batches to sample when updating."""

    discount_factor: float | dict[str, float] = 0.99
    """Parameter that balances the importance of future rewards (close to 1.0) versus immediate rewards (close to 0.0).

    Range: ``[0.0, 1.0]``.
    """

    gae_lambda: float | dict[str, float] = 0.95
    """TD(lambda) coefficient for computing Generalized Advantage Estimation (GAE)."""

    learning_rate: float | dict[str, float] = 1e-3
    """Learning rate for the value network.

    In TRPO, the policy network update uses a constrained optimization approach instead of gradient descent.
    Consequently, a learning rate is not defined for the policy network.
    """

    learning_rate_scheduler: type | None | dict[str, type | None] = None
    """Learning rate scheduler class for the value network.

    See :ref:`learning_rate_schedulers` for more details.
    """

    learning_rate_scheduler_kwargs: dict | dict[str, dict] = dataclasses.field(default_factory=dict)
    """Keyword arguments for the learning rate scheduler's constructor.

    See :ref:`learning_rate_schedulers` for more details.

    .. warning::

        The ``optimizer`` argument is automatically passed to the learning rate scheduler's constructor.
        Therefore, it must not be provided in the keyword arguments.
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

    value_preprocessor: type | None | dict[str, type | None] = None
    """Preprocessor class to process the value network's output.

    See :ref:`preprocessors` for more details.
    """

    value_preprocessor_kwargs: dict | dict[str, dict] = dataclasses.field(default_factory=dict)
    """Keyword arguments for the value preprocessor's constructor.

    See :ref:`preprocessors` for more details.
    """

    grad_norm_clip: float | dict[str, float] = 0.5
    """Clipping coefficient for the gradients by their global norm.

    If less than or equal to 0, the gradients will not be clipped.
    """

    value_clip: float | dict[str, float] = 0.2
    """Clipping coefficient for the predicted value during value loss computation.

    If less than or equal to 0, the predicted value will not be clipped.
    """

    value_loss_scale: float | dict[str, float] = 1.0
    """Value loss scaling factor."""

    damping: float | dict[str, float] = 0.1
    """Damping coefficient for computing the Hessian-vector product."""

    max_kl_divergence: float | dict[str, float] = 0.01
    """Maximum KL-divergence between the old policy and the new policy."""

    conjugate_gradient_steps: int | dict[str, int] = 10
    """Maximum number of iterations for the conjugate gradient algorithm."""

    max_backtrack_steps: int | dict[str, int] = 10
    """Maximum number of backtracking steps during line search."""

    accept_ratio: float | dict[str, float] = 0.5
    """Accept ratio for the line search loss improvement."""

    step_fraction: float | dict[str, float] = 1.0
    """Fraction of the step size for the line search."""

    time_limit_bootstrap: bool | dict[str, bool] = False
    """Whether to bootstrap at timeout termination (episode truncation)."""

    rewards_shaper: Callable | None = None
    """Rewards shaping function."""

    fixed_order: bool = False
    """Whether to update the agents' policies in a fixed order.

    * If True, the policies are updated sequentially in the order of the possible agents.
    * If False, the policies are updated sequentially in a new random order at each update.
    """

    def expand(self, *, possible_agents: list[str]) -> None:
        """Expand the configuration."""
        super().expand(possible_agents=possible_agents, immutable=["rewards_shaper", "rollouts", "fixed_order"])
