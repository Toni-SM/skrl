from typing import Union, Tuple, List

import gym
import torch
import inspect
import numpy as np


class Memory:
    def __init__(self, memory_size: int, num_envs: int = 1, device: str = "cuda:0", preallocate: bool = True) -> None:
        """
        Base class representing a memory with circular buffers

        Buffers are torch tensors with shape (memory size, number of environments, data size).
        Circular buffers are implemented with two integers: a memory index and an environment index

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
        """
        # TODO: handle dynamic memory
        # TODO: show memory consumption
        # TODO: handle advanced gym spaces

        self.preallocate = preallocate
        self.memory_size = memory_size
        self.num_envs = num_envs
        self.device = device

        self.filled = False
        self.env_index = 0
        self.memory_index = 0

    def __len__(self) -> int:
        """
        Compute and return the current (valid) size of the memory

        The valid size is calculated as the `memory_size * num_envs` if the memory is full (filled).
        Otherwise, the `memory_index * num_envs + env_index` is returned

        Returns
        -------
        int
            valid size
        """
        return self.memory_size * self.num_envs if self.filled else self.memory_index * self.num_envs + self.env_index
        
    def get_tensor_names(self) -> Tuple[str]:
        """
        Get the name of the internal tensors in alphabetical order

        Returns
        -------
        tuple of strings
            Tensor names without internal prefix (_tensor_)
        """
        names = [m[0] for m in inspect.getmembers(self, lambda name: not(inspect.isroutine(name))) if m[0].startswith('_tensor_')]
        return tuple(sorted([name[8:] for name in names]))

    def create_tensor(self, name: str, size: Union[int, Tuple[int], gym.Space], dtype: Union[torch.dtype, None] = None) -> bool:
        """
        Create a new internal tensor in memory
        
        The tensor will have a 3-components shape (memory size, number of environments, size).
        The internal representation will use _tensor_<name> as the name of the class property

        Parameters
        ----------
        name: str
            Tensor name (the name has to follow the python PEP 8 style)
        size: int, tuple or list of integers or gym.Space
            Number of elements in the last dimension (effective data size).
            The product of the elements will be computed for collections or gym spaces types
        dtype: torch.dtype or None, optional
            Data type (torch.dtype).
            If None, the global default torch data type will be used (default)

        Returns
        -------
        bool
            True if the tensor was created, otherwise False
        """
        # TODO: check memory availability
        # TODO: check existing tensor and new tensor shape
        # format tensor name to _tensor_<name>
        name = "_tensor_{}".format(name)
        # check if the tensor exists
        if hasattr(self, name):
            print("[WARNING] the tensor {} exists".format(name))
            return
        # compute data size
        if type(size) in [tuple, list]:
            size = np.prod(size)
        elif issubclass(type(size), gym.Space):
            if issubclass(type(size), gym.spaces.Discrete):
                size = size.n
            else:
                size = np.prod(size.shape)
        # create tensor
        setattr(self, name, torch.zeros((self.memory_size, self.num_envs, size), device=self.device, dtype=dtype))
        return True

    def reset(self) -> None:
        """
        Reset the memory by cleaning internal indexes and flags

        Old data will be retained until overwritten, but access through the available methods will not be guaranteed

        Default values of the internal indexes and flags
        - filled: False
        - env_index: 0
        - memory_index: 0
        """
        self.filled = False
        self.env_index = 0
        self.memory_index = 0

    def add_samples(self, **tensors: torch.Tensor) -> None:
        """
        Record samples in memory

        Samples should be a tensor with 2-components shape (number of environments, data size).
        All tensors must be of the same shape

        According to the number of environments, the following classification is made:
        - one environment:
            Store a single sample (tensors with one dimension) and increment the environment index (second index) by one
        - number of environments less than num_envs:
            Store the samples and increment the environment index (second index) by the number of the environments
        - number of environments equals num_envs:
            Store the samples and increment the memory index (first index) by one

        Parameters
        ----------
        tensors:
            Sampled data as key-value arguments where the keys are the names of the tensors to be modified.
            Non-existing tensors will be skipped
        """
        if not tensors:
            raise ValueError("No samples to be recorded in memory. Pass samples as key-value arguments (where key is the tensor name)")
        tensor = list(tensors.values())[0]
        # single environment
        # TODO: use number of environments to check one environment
        if tensor.dim() == 1:
            for name, tensor in tensors.items():
                name = "_tensor_{}".format(name)
                if hasattr(self, name):
                    getattr(self, name)[self.memory_index, self.env_index].copy_(tensor)
            self.env_index += 1
        # multi environment (number of environments less than num_envs)
        elif tensor.dim() > 1 and tensor.shape[0] < self.num_envs:
            for name, tensor in tensors.items():
                name = "_tensor_{}".format(name)
                if hasattr(self, name):
                    getattr(self, name)[self.memory_index, self.env_index:self.env_index + tensor.shape[0]].copy_(tensor if tensor.dim() == 2 else tensor.view(-1, 1))
            self.env_index += tensor.shape[0]
        # multi environment (number of environments equals num_envs)
        elif tensor.dim() > 1 and tensor.shape[0] == self.num_envs:
            for name, tensor in tensors.items():
                name = "_tensor_{}".format(name)
                if hasattr(self, name):
                    getattr(self, name)[self.memory_index].copy_(tensor if tensor.dim() == 2 else tensor.view(-1, 1))
            self.memory_index += 1

        # update indexes and flags
        if self.env_index >= self.num_envs:
            self.env_index = 0
            self.memory_index += 1
        if self.memory_index >= self.memory_size:
            self.memory_index = 0
            self.filled = True

    def sample(self, batch_size: int, names: Tuple[str]) -> Tuple[torch.Tensor]:
        """
        Data sampling method to be implemented by the inheriting classes

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
        raise NotImplementedError("The sampling method (.sample()) is not implemented")

    def sample_by_index(self, indexes: Union[tuple, np.ndarray, torch.Tensor], names: Tuple[str]) -> Tuple[torch.Tensor]:
        """
        Sample data from memory according to their indexes

        Parameters
        ----------
        indexes: tuple or list, numpy.ndarray or torch.Tensor
            Indexes used for sampling
        names: tuple or list of strings
            Tensors names from which to obtain the samples

        Returns
        -------
        list of torch.Tensor
            Sampled data from tensors sorted according to their position in the list of names.
            The sampled tensors will have the following shape: (number of indexes, data size)
        """
        # TODO: skip invalid names
        tensors = [getattr(self, "_tensor_{}".format(name)) for name in names]
        return [tensor.view(-1, tensor.size(-1))[indexes] for tensor in tensors]
