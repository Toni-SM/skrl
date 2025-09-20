from typing import Callable

import dataclasses

from skrl.agents.torch import AgentCfg


@dataclasses.dataclass(kw_only=True)
class TRPO_CFG(AgentCfg):
    """Configuration for the TRPO agent."""

    rollouts: int = 16
    """Number of collection steps to perform between updates."""

    learning_epochs: int = 8
    """Number of learning epochs to perform during updates."""

    mini_batches: int = 2
    """Number of mini batches to sample when updating."""

    discount_factor: float = 0.99
    """Parameter that balances the importance of future rewards (close to 1.0) versus immediate rewards (close to 0.0).

    Range: ``[0.0, 1.0]``.
    """

    lambda_: float = 0.95
    """TD(lambda) coefficient for computing Generalized Advantage Estimation (GAE)."""

    learning_rate: float = 1e-3
    """Learning rate for the value network.

    In TRPO, the policy network update uses a constrained optimization approach instead of gradient descent.
    Consequently, a learning rate is not defined for the policy network.
    """

    learning_rate_scheduler: type | None = None
    """Learning rate scheduler class for the value network.

    See :ref:`learning_rate_schedulers` for more details.
    """

    learning_rate_scheduler_kwargs: dict = dataclasses.field(default_factory=dict)
    """Keyword arguments for the learning rate scheduler's constructor.

    See :ref:`learning_rate_schedulers` for more details.

    .. warning::

        The ``optimizer`` argument is automatically passed to the learning rate scheduler's constructor.
        Therefore, it must not be provided in the keyword arguments.
    """

    observation_preprocessor: type | None = None
    """Preprocessor class to process the environment's observations.

    See :ref:`preprocessors` for more details.
    """

    observation_preprocessor_kwargs: dict = dataclasses.field(default_factory=dict)
    """Keyword arguments for the observation preprocessor's constructor.

    See :ref:`preprocessors` for more details.
    """

    state_preprocessor: type | None = None
    """Preprocessor class to process the environment's states.

    See :ref:`preprocessors` for more details.
    """

    state_preprocessor_kwargs: dict = dataclasses.field(default_factory=dict)
    """Keyword arguments for the state preprocessor's constructor.

    See :ref:`preprocessors` for more details.
    """

    value_preprocessor: type | None = None
    """Preprocessor class to process the value network's output.

    See :ref:`preprocessors` for more details.
    """

    value_preprocessor_kwargs: dict = dataclasses.field(default_factory=dict)
    """Keyword arguments for the value preprocessor's constructor.

    See :ref:`preprocessors` for more details.
    """

    random_timesteps: int = 0
    """Number of random exploration (sampling random actions) steps to perform before sampling actions from the policy."""

    learning_starts: int = 0
    """Number of steps to perform before calling the algorithm update function."""

    grad_norm_clip: float = 0.5
    """Clipping coefficient for the gradients by their global norm.

    If less than or equal to 0, the gradients will not be clipped.
    """

    value_loss_scale: float = 1.0
    """Value loss scaling factor."""

    damping: float = 0.1
    """Damping coefficient for computing the Hessian-vector product."""

    max_kl_divergence: float = 0.01
    """Maximum KL-divergence between the old policy and the new policy."""

    conjugate_gradient_steps: int = 10
    """Maximum number of iterations for the conjugate gradient algorithm."""

    max_backtrack_steps: int = 10
    """Maximum number of backtracking steps during line search."""

    accept_ratio: float = 0.5
    """Accept ratio for the line search loss improvement."""

    step_fraction: float = 1.0
    """Fraction of the step size for the line search."""

    time_limit_bootstrap: bool = False
    """Whether to bootstrap at timeout termination (episode truncation)."""

    rewards_shaper: Callable | None = None
    """Rewards shaping function."""
