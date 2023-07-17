from typing import List, Optional, Tuple

import jax
import numpy as np

from skrl.memories.jax import Memory


class RandomMemory(Memory):
    def __init__(self,
                 memory_size: int,
                 num_envs: int = 1,
                 device: Optional[jax.Device] = None,
                 export: bool = False,
                 export_format: str = "pt",
                 export_directory: str = "",
                 replacement=True) -> None:
        """Random sampling memory

        Sample a batch from memory randomly

        :param memory_size: Maximum number of elements in the first dimension of each internal storage
        :type memory_size: int
        :param num_envs: Number of parallel environments (default: ``1``)
        :type num_envs: int, optional
        :param device: Device on which an array is or will be allocated (default: ``None``)
        :type device: jax.Device, optional
        :param export: Export the memory to a file (default: ``False``).
                       If True, the memory will be exported when the memory is filled
        :type export: bool, optional
        :param export_format: Export format (default: ``"pt"``).
                              Supported formats: torch (pt), numpy (np), comma separated values (csv)
        :type export_format: str, optional
        :param export_directory: Directory where the memory will be exported (default: ``""``).
                                 If empty, the agent's experiment directory will be used
        :type export_directory: str, optional
        :param replacement: Flag to indicate whether the sample is with or without replacement (default: ``True``).
                            Replacement implies that a value can be selected multiple times (the batch size is always guaranteed).
                            Sampling without replacement will return a batch of maximum memory size if the memory size is less than the requested batch size
        :type replacement: bool, optional

        :raises ValueError: The export format is not supported
        """
        super().__init__(memory_size, num_envs, device, export, export_format, export_directory)

        self._replacement = replacement

    def sample(self, names: Tuple[str], batch_size: int, mini_batches: int = 1) -> List[List[jax.Array]]:
        """Sample a batch from memory randomly

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param batch_size: Number of element to sample
        :type batch_size: int
        :param mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional

        :return: Sampled data from tensors sorted according to their position in the list of names.
                 The sampled tensors will have the following shape: (batch size, data size)
        :rtype: list of jax.Array list
        """
        # generate random indexes
        if self._replacement:
            indexes = np.random.randint(0, len(self), (batch_size,))
        else:
            indexes = np.random.permutation(len(self))[:batch_size]

        return self.sample_by_index(names=names, indexes=indexes, mini_batches=mini_batches)
