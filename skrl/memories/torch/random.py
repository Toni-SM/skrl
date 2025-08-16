from typing import List, Literal, Optional, Sequence, Union

import torch

from skrl.memories.torch import Memory


class RandomMemory(Memory):
    def __init__(
        self,
        *,
        memory_size: int,
        num_envs: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        export: bool = False,
        export_format: Literal["pt", "npz", "csv"] = "pt",
        export_directory: str = "",
        replacement: bool = True,
    ) -> None:
        """Random sampling memory (sample a batch from memory randomly).

        :param memory_size: Maximum number of elements in the first dimension for each tensor.
        :param num_envs: Number of parallel environments.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param export: Export the memory to a file. If ``True``, the memory will be exported once it is filled
            and before the circular buffer starts to overwrite the oldest data.
        :param export_format: File format to export the memory.
            Supported formats: PyTorch (``"pt"``), NumPy (``"npz"``) or comma separated values (``"csv"``).
        :param export_directory: Directory where the memory files will be exported.
            If not specified, the agent's experiment directory will be used.
        :param replacement: Flag to indicate whether the sample is with or without replacement.
            Replacement implies that a value can be selected multiple times (the batch size is always guaranteed).
            Sampling without replacement will return a batch of maximum memory size if the memory size is less than
            the requested batch size.

        :raises ValueError: Unsupported export format.
        """
        super().__init__(
            memory_size=memory_size,
            num_envs=num_envs,
            device=device,
            export=export,
            export_format=export_format,
            export_directory=export_directory,
        )

        self._replacement = replacement

    def sample(
        self, names: Sequence[str], *, batch_size: int, mini_batches: int = 1, sequence_length: int = 1
    ) -> List[List[torch.Tensor]]:
        """Sample a batch from memory randomly.

        :param names: Tensors names from which to obtain the samples.
        :param batch_size: Number of elements to sample.
        :param mini_batches: Number of mini-batches to sample.
        :param sequence_length: Length of each sequence.

        :return: Sampled data from tensors sorted according to their position in the list of names.
            The sampled tensors will have the following shape: ``(batch_size, data_size)``.
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
        # sample by indexes
        self.sampling_indexes = indexes
        return self.sample_by_index(names=names, indexes=indexes, mini_batches=mini_batches)
