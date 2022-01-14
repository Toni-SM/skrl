from typing import Union, Tuple, List

import torch

from .base import Memory


class RandomMemory(Memory):
    def __init__(self, memory_size: int, num_envs: int = 1, device: Union[str, torch.device] = "cuda:0", preallocate: bool = True, replacement=True) -> None:
        """Random sampling memory

        Sample a batch from memory randomly

        :param memory_size: Maximum number of elements in the first dimension of each internal storage
        :type memory_size: int
        :param num_envs: Number of parallel environments (default: 1)
        :type num_envs: int, optional
        :param device: Device on which a torch tensor is or will be allocated (default: "cuda:0")
        :type device: str or torch.device, optional
        :param preallocate: If true, preallocate memory for efficient use (default: True)
        :type preallocate: bool, optional
        :param replacement: Flag to indicate whether the sample is with or without replacement (default: True). 
                            Replacement implies that a value can be selected multiple times (the batch size is always guaranteed).
                            Sampling without replacement will return a batch of maximum memory size if the memory size is less than the requested batch size
        :type replacement: bool, optional
        """
        super().__init__(memory_size, num_envs, device, preallocate)

        self._replacement = replacement

    def sample(self, names: Tuple[str], batch_size: int, mini_batches: int = 1) -> List[List[torch.Tensor]]:
        """Sample a batch from memory randomly

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param batch_size: Number of element to sample
        :type batch_size: int
        :param mini_batches: Number of mini-batches to sample (default: 1)
        :type mini_batches: int, optional

        :return: Sampled data from tensors sorted according to their position in the list of names.
                 The sampled tensors will have the following shape: (batch size, data size)
        :rtype: list of torch.Tensor list
        """
        # generate random indexes
        if self._replacement:
            indexes = torch.randint(0, len(self), (batch_size,), device=self.device)
        else:
            # details about the random sampling performance can be found here: 
            # https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/19
            indexes = torch.randperm(len(self), dtype=torch.long, device=self.device)[:batch_size]

        return self.sample_by_index(names=names, indexes=indexes, mini_batches=mini_batches)
