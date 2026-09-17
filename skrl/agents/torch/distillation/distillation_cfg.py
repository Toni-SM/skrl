from __future__ import annotations

from typing import Literal

import dataclasses

from skrl.agents.torch import AgentCfg


@dataclasses.dataclass(kw_only=True)
class DISTILLATION_CFG(AgentCfg):
    """Configuration for the Teacher-Student Distillation agent.

    .. note::

        The agent's ``policy`` model is the **student**, the model that acts in the environment and the only one
        whose parameters are optimized. The ``teacher`` model is frozen and kept in evaluation mode.
    """

    rollouts: int = 16
    """Number of collection steps to perform between updates."""

    learning_epochs: int = 1
    """Number of learning epochs to perform during updates."""

    mini_batches: int = 1
    """Number of mini batches to sample when updating."""

    learning_rate: float = 1e-3
    """Learning rate for the student (policy) network."""

    learning_rate_scheduler: type | None = None
    """Learning rate scheduler class for the student (policy) network.

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

    grad_norm_clip: float = 0.5
    """Clipping coefficient for the gradients by their global norm.

    If less than or equal to 0, the gradients will not be clipped.
    """

    loss_type: Literal["mse", "huber"] = "mse"
    """Regression loss used to fit the student's actions to the teacher's actions."""

    huber_delta: float = 1.0
    """Threshold at which the Huber loss changes between its delta-scaled L1 and L2 behavior.

    Only used if ``loss_type`` is set to ``"huber"``.
    """

    teacher_checkpoint: str = ""
    """Path to the checkpoint file from which to load the **teacher** model's parameters.

    The referenced file can be either a whole-agent checkpoint (e.g.: ``agent_48000.pt``), in which case the
    ``teacher`` or the ``policy`` module is used (in that order), or a single model's state dict
    (e.g.: ``policy_48000.pt``), as written when the ``experiment.store_separately`` option is enabled.

    If empty, the teacher model is expected to be loaded by the user, either by calling the agent's
    :py:meth:`~skrl.agents.torch.distillation.Distillation.load_teacher` method or by loading the model directly.
    """

    mixed_precision: bool = False
    """Whether to enable automatic mixed precision for higher performance."""

    def validate(self) -> bool:
        """Validate the configuration.

        :raises ValueError: If the ``loss_type`` is not supported.

        :return: True if the configuration is valid.
        """
        super().validate()
        if self.loss_type not in ["mse", "huber"]:
            raise ValueError(f"Unsupported 'loss_type': {self.loss_type}. Supported types are: ['mse', 'huber']")
        return True
