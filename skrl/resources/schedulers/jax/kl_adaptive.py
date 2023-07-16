from typing import Optional, Union

import numpy as np


class KLAdaptiveLR:
    def __init__(self,
                 init_value: float,
                 kl_threshold: float = 0.008,
                 min_lr: float = 1e-6,
                 max_lr: float = 1e-2,
                 kl_factor: float = 2,
                 lr_factor: float = 1.5) -> None:
        """Adaptive KL scheduler

        Adjusts the learning rate according to the KL divergence.
        The implementation is adapted from the rl_games library
        (https://github.com/Denys88/rl_games/blob/master/rl_games/common/schedulers.py)

        .. note::

            This scheduler is only available for PPO at the moment.
            Applying it to other agents will not change the learning rate

        Example::

            >>> scheduler = KLAdaptiveLR(init_value=1e-3, kl_threshold=0.01)
            >>> for epoch in range(100):
            >>>     # ...
            >>>     kl_divergence = ...
            >>>     scheduler.step(kl_divergence)
            >>>     scheduler.lr  # get the updated learning rate

        :param init_value: Initial learning rate
        :type init_value: float
        :param kl_threshold: Threshold for KL divergence (default: ``0.008``)
        :type kl_threshold: float, optional
        :param min_lr: Lower bound for learning rate (default: ``1e-6``)
        :type min_lr: float, optional
        :param max_lr: Upper bound for learning rate (default: ``1e-2``)
        :type max_lr: float, optional
        :param kl_factor: The number used to modify the KL divergence threshold (default: ``2``)
        :type kl_factor: float, optional
        :param lr_factor: The number used to modify the learning rate (default: ``1.5``)
        :type lr_factor: float, optional
        """
        self.kl_threshold = kl_threshold
        self.min_lr = min_lr
        self.max_lr = max_lr
        self._kl_factor = kl_factor
        self._lr_factor = lr_factor

        self._lr = init_value

    @property
    def lr(self) -> float:
        """Learning rate
        """
        return self._lr

    def step(self, kl: Optional[Union[np.ndarray, float]] = None) -> None:
        """
        Step scheduler

        Example::

            >>> kl = [0.0332, 0.0500, 0.0383, 0.0456, 0.0076, 0.0240, 0.0164]
            >>> kl
            [0.0332, 0.05, 0.0383, 0.0456, 0.0076, 0.024, 0.0164]
            >>> scheduler.step(np.mean(kl))

            >>> kl = 0.0046
            >>> scheduler.step(kl)

        :param kl: KL divergence (default: ``None``)
                   If None, no adjustment is made.
                   If array, the number of elements must be 1
        :type kl: np.ndarray, float or None, optional
        """
        if kl is not None:
            if kl > self.kl_threshold * self._kl_factor:
                self._lr = max(self._lr / self._lr_factor, self.min_lr)
            elif kl < self.kl_threshold / self._kl_factor:
                self._lr = min(self._lr * self._lr_factor, self.max_lr)


# Alias to maintain naming compatibility with Optax schedulers
# https://optax.readthedocs.io/en/latest/api.html#schedules
kl_adaptive = KLAdaptiveLR
