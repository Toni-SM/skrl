from typing import Tuple

import torch
import numpy as np

from .base import Memory


class PrioritizedMemory(Memory):
    def __init__(self, memory_size: int, num_envs: int = 1, device: str = "cuda:0", preallocate: bool = True, alpha: float = 0.5, beta: float = 0.4, eps: float = 1e-6) -> None:
        """
        Prioritized sampling memory

        Sample a batch from memory randomly

        Parameters
        ----------
        memory_size: int
            Maximum number of elements in the first dimension of each internal storage
        num_envs: int
            Number of parallel environments
        device: str, optional
            Device on which a torch tensor is or will be allocated (default: "cuda:0")
        preallocate: bool, optional
            If true, preallocate memory for efficient use (default: True)
        alpha: float, optional
            Hyperparameter for prioritized sampling (default: 0.5)
        beta: float, optional
            Hyperparameter for prioritized sampling (default: 0.4)
        eps: float, optional
            Hyperparameter for prioritized sampling (default: 1e-6)
        """
        super().__init__(memory_size, num_envs, device, preallocate)

        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def sample(self, batch_size: int, names: Tuple[str]) -> Tuple[torch.Tensor]:
        """
        Sample a batch from memmory randomly
        
        Parameters
        ----------
        batch_size: int
            Number of element to sample
        names: tuple or list of strings
            Tensors names from which to obtain the samples

        Returns
        -------
        tuple of torch.Tensor
            Sampled data from tensors sorted according to their position in the list of names.
            The sampled tensors will have the following shape: (batch size, data size)
        """
        # generate random indexes
        if self._replacement:
            indexes = np.random.choice(len(self), size=batch_size, replace=True)
        else:
            indexes = np.random.choice(len(self), size=min([batch_size, len(self)]), replace=False)
        
        return self.sample_by_index(indexes=indexes, names=names)
