from typing import Optional, Union

import numpy as np
import optax


def KLAdaptiveLR(
    *,
    kl_threshold: float = 0.008,
    min_lr: float = 1e-6,
    max_lr: float = 1e-2,
    kl_factor: float = 2,
    lr_factor: float = 1.5,
) -> optax.Schedule:
    """Adaptive KL scheduler.

    Adjusts the learning rate according to the KL divergence.

    .. note::

        This scheduler is only available for the A2C, AMP, PPO and RPO single-agent algorithms,
        and IPPO and MAPPO multi-agent algorithms. Applying it to other agents will not change the learning rate.

    :param kl_threshold: Threshold for KL divergence.
    :param min_lr: Lower bound for learning rate.
    :param max_lr: Upper bound for learning rate.
    :param kl_factor: The number used to modify the KL divergence threshold.
    :param lr_factor: The number used to modify the learning rate.

    :return: A function that maps step counts, current learning rate and KL divergence to the new learning rate value.
        If no learning rate is specified, 1.0 will be returned to mimic the Optax's scheduler behaviors.
        If the learning rate is specified but the KL divergence is not 0, the specified learning rate is returned.

    Example::

        >>> scheduler = KLAdaptiveLR(kl_threshold=0.01)
        >>> for epoch in range(100):
        >>>     # ...
        >>>     kl_divergence = ...
        >>>     new_lr = scheduler(timestep, lr, kl_divergence)
    """

    def schedule(count: int, *, lr: Optional[float] = None, kl: Optional[Union[np.ndarray, float]] = None) -> float:
        if lr is None:
            return 1.0
        if kl is not None:
            if kl > kl_threshold * kl_factor:
                lr = max(lr / lr_factor, min_lr)
            elif kl < kl_threshold / kl_factor:
                lr = min(lr * lr_factor, max_lr)
        return lr

    return schedule


# Alias to maintain naming compatibility with Optax schedulers
# https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html
kl_adaptive = KLAdaptiveLR
