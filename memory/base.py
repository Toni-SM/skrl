from typing import Union

import gym
import math
import torch
import inspect
import numpy as np


class Memory:
    def __init__(self, memory_size: int, num_envs: int = 1, device: str = "cuda:0", preallocate: bool = True) -> None:
        """
        Base class that represent a memory with circular buffers

        Parameters
        ----------
        memory_size: int
            Maximum number of elements in the first dimension of each internal tensor
        num_envs: int
            Number of parallel environments
        device: str
            Device on which a PyTorch tensor is or will be allocated
        preallocate: bool
            If true, preallocate memory for efficient use
        state_space: 
            State/observation space
        action_space: gym.Space or None
            Action space
        """
        # TODO: handle dynamic memory
        # TODO: show memory consumption
        # TODO: handle advanced gym spaces

        self.preallocate = preallocate
        self.memory_size = memory_size
        self.num_envs = num_envs
        self.device = device

        self.filled = False
        self.pointer_env = 0
        self.pointer_memory = 0

        # alternative names
        self.push = self.add_samples
        self.add_sample = self.add_samples

    def __len__(self) -> int:
        """
        Compute and return the current (valid) size of the memory

        The valid size is calculated as the memory size * number of environments if the memory is full (filled).
        Otherwise, the memory pointer * number of environments + environment pointer is returned

        Returns
        -------
        int
            valid size
        """
        return self.memory_size * self.num_envs if self.filled else self.pointer_memory * self.num_envs + self.pointer_env
        
    def get_tensor_names(self) -> list[str]:
        """
        Get the name of the internal tensors in alphabetical order

        Returns
        -------
        list of strings
            Tensor names without internal prefix (_tensor_)
        """
        names = [member[0] for member in inspect.getmembers(self, lambda name: not(inspect.isroutine(name))) if member[0].startswith('_tensor_')]
        return sorted([name for name in names])

    def create_tensor(self, name: str, size: Union[int, tuple[int], list[int], gym.Space], dtype: Union[torch.dtype, None] = None) -> bool:
        """
        Create a new internal tensor in the memory with a 3-component shape (memory_size, num_envs, size)

        Parameters
        ----------
        name: str:
            Tensor name. 
            The internal representation will use _tensor_<name>
        size: int, tuple, list or gym.Space
            Number of elements in the last dimension (effective data size)
            The product of the elements will be computed for collections or gym spaces types
        dtype: torch.dtype
            Data type (torch.dtype)

        Returns
        -------
        bool
            True if the tensor was created, otherwise False
        """
        # TODO: check memory availability
        # format tensor name to _tensor_<name>
        name = name if name.startswith("_tensor_") else "_tensor_{}".format(name)
        # check if the tensor exists
        if hasattr(self, name):
            print("[WARNING] the tensor {} exists".format(name))
            return
        # compute data size
        if type(size) in [tuple, list]:
            size = math.prod(size)
        elif issubclass(type(size), gym.Space):
            size = math.prod(size.shape)
        # create tensor
        setattr(self, name, torch.zeros((self.memory_size, self.num_envs, size), device=self.device, dtype=dtype))
        return True

    def reset(self) -> None:
        """
        Reset the memory by cleaning internal flags

        Reset values of the flags
        - filled: False
        - pointer_env: 0
        - pointer_memory: 0

        The old data will be kept until they are overwritten, however their access through the available methods will not be guaranteed
        """
        self.filled = False
        self.pointer_env = 0
        self.pointer_memory = 0

    def add_samples(self, **tensors: torch.Tensor) -> None:
        """
        Record samples with 2-component shape (num_envs, size) in the memory

        According to the number of environments, the following classification is made:
        - one environment:
            Store a single sample (tensors with one dimension) and increment the environment pointer (second index) by one
        - number of environments less than num_envs:
            Store the samples and increment the environment pointer (second index) by the number of the environments
        - number of environments equals num_envs:
            Store the samples and increment the memory pointer (first index) by one

        Parameters
        ----------
        tensors:
            Sampled data where the keys are the names of the tensors to be modified.
            Non-existing tensors will be skipped
        """
        if not tensors:
            raise ValueError("No samples to be recorded in memory. Pass samples as named arguments (keyword containing the tensor name)")
        tensor = list(tensors.values())[0]
        # single environment
        if tensor.dim() == 1:
            for name, tensor in tensors.items():
                name = name if name.startswith("_tensor_") else "_tensor_{}".format(name)
                if hasattr(self, name):
                    getattr(self, name)[self.pointer_memory, self.pointer_env].copy_(tensor)
            self.pointer_env += 1
        # multi environment (where the amount of environments is equal to num_envs)
        elif tensor.dim() > 1 and tensor.shape[0] == self.num_envs:
            for name, tensor in tensors.items():
                name = name if name.startswith("_tensor_") else "_tensor_{}".format(name)
                if hasattr(self, name):
                    getattr(self, name)[self.pointer_memory].copy_(tensor)
            self.pointer_memory += 1
        # multi environment (where the amount of environments is less than the num_envs)
        elif tensor.dim() > 1 and tensor.shape[0] < self.num_envs:
            for name, tensor in tensors.items():
                name = name if name.startswith("_tensor_") else "_tensor_{}".format(name)
                if hasattr(self, name):
                    getattr(self, name)[self.pointer_memory, self.pointer_env:self.pointer_env + tensor.shape[0]].copy_(tensor)
            self.pointer_env += tensor.shape[0]

        # update pointers
        if self.pointer_env >= self.num_envs:
            self.pointer_env = 0
            self.pointer_memory += 1
        if self.pointer_memory >= self.memory_size:
            self.pointer_memory = 0
            self.filled = True

    def sample(self, batch_size: int, names: list[str] = []) -> list[torch.Tensor]:
        """
        Sample a batch of data from memory

        Parameters
        ----------
        batch_size: int
            Number of element to sample
        names: list of strings
            List of tensors names from which to obtain the samples

        Returns
        -------
        list of torch.Tensor
            Sampled data from tensors sorted according to their position in the list of names
        """
        raise NotImplementedError("The sampling method (.sample()) is not implemented")

    def sample_by_index(self, indexes: Union[list, np.ndarray, torch.Tensor], names: list[str]) -> list[torch.Tensor]:
        """
        Sample data from memory according to its indexes

        Parameters
        ----------
        indexes: list, numpy.ndarray or torch.Tensor
            Indexes
        names: list of strings
            List of tensors names from which to obtain the samples

        Returns
        -------
        list of torch.Tensor
            Sampled data from tensors sorted according to their position in the list of names 
        """
        tensors = [getattr(self, name) for name in names]
        return [tensor.view(-1, tensor.size(-1))[indexes] for tensor in tensors]
