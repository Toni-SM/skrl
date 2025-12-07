from __future__ import annotations

from typing import Callable

import dataclasses

from skrl.agents.torch import AgentCfg


@dataclasses.dataclass(kw_only=True)
class AMP_CFG(AgentCfg):
    """Configuration for the AMP agent."""

    rollouts: int = 16
    """Number of collection steps to perform between updates."""

    learning_epochs: int = 6
    """Number of learning epochs to perform during updates."""

    mini_batches: int = 2
    """Number of mini batches to sample when updating."""

    discount_factor: float = 0.99
    """Parameter that balances the importance of future rewards (close to 1.0) versus immediate rewards (close to 0.0).

    Range: ``[0.0, 1.0]``.
    """

    lambda_: float = 0.95
    """TD(lambda) coefficient for computing Generalized Advantage Estimation (GAE)."""

    learning_rate: float | tuple[float, float, float] = 5e-5
    """Learning rate for the policy, value and discriminator networks.

    * If a float is provided, the same learning rate will be used for the networks.
    * If a tuple is provided, its elements will be used for each network in order.
    """

    learning_rate_scheduler: type | tuple[type | None, type | None, type | None] | None = None
    """Learning rate scheduler class for the policy, value and discriminator networks.

    See :ref:`learning_rate_schedulers` for more details.

    * If a class is provided, the same learning rate scheduler will be used for the networks.
    * If a tuple is provided, its elements will be used for each network in order.
    """

    learning_rate_scheduler_kwargs: dict | tuple[dict, dict, dict] = dataclasses.field(default_factory=dict)
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

    value_preprocessor: type | None = None
    """Preprocessor class to process the value network's output.

    See :ref:`preprocessors` for more details.
    """

    value_preprocessor_kwargs: dict = dataclasses.field(default_factory=dict)
    """Keyword arguments for the value preprocessor's constructor.

    See :ref:`preprocessors` for more details.
    """

    amp_observation_preprocessor: type | None = None
    """Preprocessor class to process the environment's AMP observations.

    See :ref:`preprocessors` for more details.
    """

    amp_observation_preprocessor_kwargs: dict = dataclasses.field(default_factory=dict)
    """Keyword arguments for the AMP observation preprocessor's constructor.

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

    ratio_clip: float = 0.2
    """Clipping coefficient for computing the clipped surrogate objective."""

    value_clip: float = 0.2
    """Clipping coefficient for the predicted value during value loss computation.

    If less than or equal to 0, the predicted value will not be clipped.
    """

    entropy_loss_scale: float = 0.0
    """Entropy loss scaling factor."""

    value_loss_scale: float = 2.5
    """Value loss scaling factor."""

    discriminator_loss_scale: float = 5.0
    """Discriminator loss scaling factor."""

    amp_batch_size: int = 512
    """Batch size for updating the reference motion dataset."""

    task_reward_scale: float = 0.0
    """Reward scaling factor for the task."""

    style_reward_scale: float = 2.0
    """Reward scaling factor for the style (motion to be copied)."""

    discriminator_batch_size: int = -1
    """Batch size (for subsampling AMP observations) for computing the discriminator loss.

    If less than or equal to 0, all sampled AMP observations will be used.
    """

    discriminator_logit_regularization_scale: float = 0.05
    """Logit regularization scaling factor for the discriminator."""

    discriminator_gradient_penalty_scale: float = 5.0
    """Gradient penalty scaling factor for the discriminator."""

    discriminator_weight_decay_scale: float = 0.0001
    """Weight decay scaling factor for the discriminator."""

    time_limit_bootstrap: bool = False
    """Whether to bootstrap at timeout termination (episode truncation)."""

    rewards_shaper: Callable | None = None
    """Rewards shaping function."""

    mixed_precision: bool = False
    """Whether to enable automatic mixed precision for higher performance."""

    def expand(self) -> None:
        """Expand the configuration."""
        super().expand()
        # learning rate
        if not isinstance(self.learning_rate, (tuple, list)):
            self.learning_rate = (self.learning_rate, self.learning_rate, self.learning_rate)
        # learning rate scheduler
        if self.learning_rate_scheduler is None:
            self.learning_rate_scheduler = (None, None, None)
        elif not isinstance(self.learning_rate_scheduler, (tuple, list)):
            self.learning_rate_scheduler = (
                self.learning_rate_scheduler,
                self.learning_rate_scheduler,
                self.learning_rate_scheduler,
            )
        # learning rate scheduler kwargs
        if not isinstance(self.learning_rate_scheduler_kwargs, (tuple, list)):
            self.learning_rate_scheduler_kwargs = (
                self.learning_rate_scheduler_kwargs,
                self.learning_rate_scheduler_kwargs,
                self.learning_rate_scheduler_kwargs,
            )
