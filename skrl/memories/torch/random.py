from typing import List, Optional, Tuple, Union

import torch

from skrl.memories.torch import Memory


class RandomMemory(Memory):
    def __init__(self,
                 memory_size: int,
                 num_envs: int = 1,
                 device: Optional[Union[str, torch.device]] = None,
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
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
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

    def sample(self,
               names: Tuple[str],
               batch_size: int,
               mini_batches: int = 1,
               sequence_length: int = 1) -> List[List[torch.Tensor]]:
        """Sample a batch from memory randomly

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param batch_size: Number of element to sample
        :type batch_size: int
        :param mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional
        :param sequence_length: Length of each sequence (default: ``1``)
        :type sequence_length: int, optional

        :return: Sampled data from tensors sorted according to their position in the list of names.
                 The sampled tensors will have the following shape: (batch size, data size)
        :rtype: list of torch.Tensor list
        """
        # compute valid memory sizes
        size = len(self)
        if sequence_length > 1:
            sequence_indexes = torch.arange(0, self.num_envs * sequence_length, self.num_envs)
            size -= sequence_indexes[-1].item()

        # generate random indexes
        if self._replacement:
            indexes = torch.randint(0, size, (batch_size,))
        else:
            # details about the random sampling performance can be found here:
            # https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/19
            indexes = torch.randperm(size, dtype=torch.long)[:batch_size]

        # generate sequence indexes
        if sequence_length > 1:
            indexes = (sequence_indexes.repeat(indexes.shape[0], 1) + indexes.view(-1, 1)).view(-1)

        self.sampling_indexes = indexes
        return self.sample_by_index(names=names, indexes=indexes, mini_batches=mini_batches)
