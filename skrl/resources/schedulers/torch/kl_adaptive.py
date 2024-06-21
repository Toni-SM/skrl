from typing import Optional, Union

from packaging import version

import torch
from torch.optim.lr_scheduler import _LRScheduler

from skrl import config


class KLAdaptiveLR(_LRScheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 kl_threshold: float = 0.008,
                 min_lr: float = 1e-6,
                 max_lr: float = 1e-2,
                 kl_factor: float = 2,
                 lr_factor: float = 1.5,
                 last_epoch: int = -1,
                 verbose: bool = False) -> None:
        """Adaptive KL scheduler

        Adjusts the learning rate according to the KL divergence.
        The implementation is adapted from the rl_games library
        (https://github.com/Denys88/rl_games/blob/master/rl_games/common/schedulers.py)

        .. note::

            This scheduler is only available for PPO at the moment.
            Applying it to other agents will not change the learning rate

        .. note::

            In distributed runs, the learning rate will be reduced and broadcasted across all workers/processes

        Example::

            >>> scheduler = KLAdaptiveLR(optimizer, kl_threshold=0.01)
            >>> for epoch in range(100):
            >>>     # ...
            >>>     kl_divergence = ...
            >>>     scheduler.step(kl_divergence)

        :param optimizer: Wrapped optimizer
        :type optimizer: torch.optim.Optimizer
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
        :param last_epoch: The index of last epoch (default: ``-1``)
        :type last_epoch: int, optional
        :param verbose: Verbose mode (default: ``False``)
        :type verbose: bool, optional
        """
        if version.parse(torch.__version__) >= version.parse("2.2"):
            verbose = "deprecated"
        super().__init__(optimizer, last_epoch, verbose)

        self.kl_threshold = kl_threshold
        self.min_lr = min_lr
        self.max_lr = max_lr
        self._kl_factor = kl_factor
        self._lr_factor = lr_factor

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def step(self, kl: Optional[Union[torch.Tensor, float]] = None, epoch: Optional[int] = None) -> None:
        """
        Step scheduler

        Example::

            >>> kl = torch.distributions.kl_divergence(p, q)
            >>> kl
            tensor([0.0332, 0.0500, 0.0383,  ..., 0.0076, 0.0240, 0.0164])
            >>> scheduler.step(kl.mean())

            >>> kl = 0.0046
            >>> scheduler.step(kl)

        :param kl: KL divergence (default: ``None``)
                   If None, no adjustment is made.
                   If tensor, the number of elements must be 1
        :type kl: torch.Tensor, float or None, optional
        :param epoch: Epoch (default: ``None``)
        :type epoch: int, optional
        """
        if kl is not None:
            # reduce (collect from all workers/processes) learning rate in distributed runs
            if config.torch.is_distributed:
                torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                kl /= config.torch.world_size

            for i, group in enumerate(self.optimizer.param_groups):
                # adjust the learning rate
                lr = group['lr']
                if kl > self.kl_threshold * self._kl_factor:
                    lr = max(lr / self._lr_factor, self.min_lr)
                elif kl < self.kl_threshold / self._kl_factor:
                    lr = min(lr * self._lr_factor, self.max_lr)

                # broadcast learning rate in distributed runs
                if config.torch.is_distributed:
                    lr_tensor = torch.tensor([lr], device=config.torch.device)
                    torch.distributed.broadcast(lr_tensor, 0)
                    lr = lr_tensor.item()

                # update value
                group['lr'] = lr
                self._last_lr[i] = lr
