from __future__ import annotations

from typing import Callable

import dataclasses

from skrl.agents.jax import AgentCfg


@dataclasses.dataclass(kw_only=True)
class DDQN_CFG(AgentCfg):
    """Configuration for the DDQN agent."""

    gradient_steps: int = 1
    """Number of gradient steps to perform for each update."""

    batch_size: int = 64
    """Batch size for sampling transitions from memory during training."""

    discount_factor: float = 0.99
    """Parameter that balances the importance of future rewards (close to 1.0) versus immediate rewards (close to 0.0).

    Range: ``[0.0, 1.0]``.
    """

    polyak: float = 0.005
    """Parameter to control the update of the target networks by polyak averaging.

    Range: ``[0.0, 1.0]``. See :py:meth:`~skrl.models.jax.base.Model.update_parameters` for more details.
    """

    learning_rate: float = 1e-3
    """Learning rate for the Q-network."""

    learning_rate_scheduler: type | None = None
    """Learning rate scheduler class for the Q-network.

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

    random_timesteps: int = 0
    """Number of random exploration (sampling random actions) steps to perform before sampling actions from the policy."""

    learning_starts: int = 0
    """Number of steps to perform before calling the algorithm update function."""

    update_interval: int = 1
    """Number of environment steps to perform between algorithm updates."""

    target_update_interval: int = 10
    """Number of algorithm updates to perform between target network updates."""

    exploration_scheduler: Callable[[int, int], float] | None = None
    """Epsilon-greedy exploration scheduler function.

    The function takes the current ``timestep`` and the total number of ``timesteps`` as arguments
    and returns an epsilon value used to sample greedy actions.
    """

    rewards_shaper: Callable | None = None
    """Rewards shaping function."""
