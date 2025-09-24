from typing import Callable

import dataclasses

from skrl.agents.torch import AgentCfg


@dataclasses.dataclass(kw_only=True)
class TD3_CFG(AgentCfg):
    """Configuration for the TD3 agent."""

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

    Range: ``[0.0, 1.0]``. See :py:meth:`~skrl.models.torch.base.Model.update_parameters` for more details.
    """

    learning_rate: float | tuple[float, float] = 1e-3
    """Learning rate for the actor and critic networks.

    * If a float is provided, the same learning rate will be used for the networks.
    * If a tuple is provided, its elements will be used for each network in order.
    """

    learning_rate_scheduler: type | tuple[type | None, type | None] | None = None
    """Learning rate scheduler class for the actor and critic networks.

    See :ref:`learning_rate_schedulers` for more details.

    * If a class is provided, the same learning rate scheduler will be used for the networks.
    * If a tuple is provided, its elements will be used for each network in order.
    """

    learning_rate_scheduler_kwargs: dict | tuple[dict, dict] = dataclasses.field(default_factory=dict)
    """Keyword arguments for the learning rate scheduler's constructor.

    See :ref:`learning_rate_schedulers` for more details.

    .. warning::

        The ``optimizer`` argument is automatically passed to the learning rate scheduler's constructor.
        Therefore, it must not be provided in the keyword arguments.

    * If a dictionary is provided, the same keyword arguments will be used for the networks.
    * If a tuple is provided, its elements will be used for each network in order.
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

    grad_norm_clip: float = 0
    """Clipping coefficient for the gradients by their global norm.

    If less than or equal to 0, the gradients will not be clipped.
    """

    exploration_noise: type | None = None
    """Exploration noise class.

    See :ref:`noises` for more details.
    """

    exploration_noise_kwargs: dict = dataclasses.field(default_factory=dict)
    """Keyword arguments for the exploration noise's constructor.

    See :ref:`noises` for more details.
    """

    exploration_scheduler: Callable[[int, int], float] | None = None
    """Exploration scheduler function.

    The function takes the current ``timestep`` and the total number of ``timesteps`` as arguments
    and returns a scale value to apply to the sampled exploration noise.
    """

    policy_delay: int = 2
    """Policy delay update with respect to critic update."""

    smooth_regularization_noise: type | None = None
    """Smooth regularization noise class.

    See :ref:`noises` for more details.
    """

    smooth_regularization_noise_kwargs: dict = dataclasses.field(default_factory=dict)
    """Keyword arguments for the smooth regularization noise's constructor.

    See :ref:`noises` for more details.
    """

    smooth_regularization_clip: float = 0.5
    """Clipping coefficient for smooth regularization."""

    rewards_shaper: Callable | None = None
    """Rewards shaping function."""

    mixed_precision: bool = False
    """Whether to enable automatic mixed precision for higher performance."""

    def expand(self) -> None:
        """Expand the configuration."""
        super().expand()
        # learning rate
        if not isinstance(self.learning_rate, (tuple, list)):
            self.learning_rate = (self.learning_rate, self.learning_rate)
        # learning rate scheduler
        if self.learning_rate_scheduler is None:
            self.learning_rate_scheduler = (None, None)
        elif not isinstance(self.learning_rate_scheduler, (tuple, list)):
            self.learning_rate_scheduler = (self.learning_rate_scheduler, self.learning_rate_scheduler)
        # learning rate scheduler kwargs
        if not isinstance(self.learning_rate_scheduler_kwargs, (tuple, list)):
            self.learning_rate_scheduler_kwargs = (
                self.learning_rate_scheduler_kwargs,
                self.learning_rate_scheduler_kwargs,
            )
