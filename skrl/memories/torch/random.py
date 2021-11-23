from typing import Tuple

import torch
import numpy as np

from .base import Memory


class RandomMemory(Memory):
    def __init__(self, memory_size: int, num_envs: int = 1, device: str = "cuda:0", preallocate: bool = True, replacement=True) -> None:
        """Random sampling memory

        Sample a batch from memory randomly

        :param memory_size: Maximum number of elements in the first dimension of each internal storage
        :type memory_size: int
        :param num_envs: Number of parallel environments (default: 1)
        :type num_envs: int, optional
        :param device: Device on which a torch tensor is or will be allocated (default: "cuda:0")
        :type device: str, optional
        :param preallocate: If true, preallocate memory for efficient use (default: True)
        :type preallocate: bool, optional
        :param replacement: Flag to indicate whether the sample is with or without replacement (default: True). 
                            Replacement implies that a value can be selected multiple times (the batch size is always guaranteed).
                            Sampling without replacement will return a batch of maximum memory size if the memory size is less than the requested batch size
        :type replacement: bool, optional
        """
        super().__init__(memory_size, num_envs, device, preallocate)

        self._replacement = replacement

    def sample(self, batch_size: int, names: Tuple[str]) -> Tuple[torch.Tensor]:
        """Sample a batch from memory randomly

        :param batch_size: Number of element to sample
        :type batch_size: int
        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings

        :return: Sampled data from tensors sorted according to their position in the list of names.
                 The sampled tensors will have the following shape: (batch size, data size)
        :rtype: tuple of torch.Tensor
        """        
        # generate random indexes
        if self._replacement:
            indexes = np.random.choice(len(self), size=batch_size, replace=True)
        else:
            indexes = np.random.choice(len(self), size=min([batch_size, len(self)]), replace=False)
        
        return self.sample_by_index(indexes=indexes, names=names)
