from typing import Dict, Union, Tuple

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

    def sample_all(self, names: Tuple[str]) -> Tuple[torch.Tensor]:
        """
        Sample all data from memory
        # TODO: Sample only valid data

        Parameters
        ----------
        names: tuple or list of strings
            Tensors names from which to obtain the samples

        Returns
        -------
        tuple of torch.Tensor
            Sampled data from memory.
            The sampled tensors will have the following shape: (memory size * number of environments, data size)
        """
        tensors = [getattr(self, "_tensor_{}".format(name)) for name in names]
        return [tensor.view(-1, tensor.size(-1)) for tensor in tensors]

    def compute_functions(self, states_src: str = "states", actions_src: str = "actions", rewards_src: str = "rewards", next_states_src: str = "next_states", dones_src: str = "dones", values_src: str = "values", returns_dst: Union[str, None] = None, advantages_dst: Union[str, None] = None, last_values: Union[torch.Tensor, None] = None, hyperparameters: Dict = {"discount_factor": 0.99, "lambda_parameter": 0.95, "normalize_returns": False, "normalize_advantages": True}) -> None:
        """
        Compute the following functions for the given tensor names

        Available functions:
        - Returns (total discounted reward)
        - Advantages (total discounted reward - baseline)

        Parameters
        ----------
        states_src: str, optional
            Name of the tensor containing the states (default: "states")
        actions_src: str, optional
            Name of the tensor containing the actions (default: "actions")
        rewards_src: str, optional
            Name of the tensor containing the rewards (default: "rewards")
        next_states_src: str, optional
            Name of the tensor containing the next states (default: "next_states")
        dones_src: str, optional
            Name of the tensor containing the dones (default: "dones")
        values_src: str, optional
            Name of the tensor containing the values (default: "values")
        returns_dst: str or None, optional
            Name of the tensor where the returns will be stored (default: None)
        advantages_dst: str or None, optional
            Name of the tensor where the advantages will be stored (default: None)
        last_values: torch.Tensor or None, optional
            Last values (default: None).
            If None, the last values will be obtained from the tensor containing the values
        hyperparameters: dict, optional
            Hyperparameters to control the computation of the functions
            The following hyperparameters are expected:
            - discount_factor: float
                Discount factor (gamma) for the computation of the returns and the advantages (default: 0.99)
            - lambda_parameter: float
                Lambda parameter (lam) for the computation of the returns and the advantages (default: 0.95)
            - normalize_returns: bool
                If True, the returns will be normalized (default: False)
            - normalize_advantages: bool
                If True, the advantages will be normalized (default: True)
        """
        # TODO: compute functions attending the circular buffer logic
        # TODO: get last values from the last samples (if not provided) and ignore them in the computation

        # get source and destination tensors
        rewards = getattr(self, "_tensor_{}".format(rewards_src))
        dones = getattr(self, "_tensor_{}".format(dones_src))
        values = getattr(self, "_tensor_{}".format(values_src))
        
        returns = getattr(self, "_tensor_{}".format(returns_dst)) if returns_dst is not None else torch.zeros_like(rewards)
        advantages = getattr(self, "_tensor_{}".format(advantages_dst)) if advantages_dst is not None else torch.zeros_like(rewards)

        # hyperarameters
        discount_factor = hyperparameters.get("discount_factor", 0.99)
        lambda_parameter = hyperparameters.get("lambda_parameter", 0.95)
        normalize_returns = hyperparameters.get("normalize_returns", False)
        normalize_advantages = hyperparameters.get("normalize_advantages", True)

        # compute and normalize the returns
        if returns_dst is not None or advantages_dst is not None:
            advantage = 0
            for step in reversed(range(self.memory_size)):
                next_values = values[step + 1] if step < self.memory_size - 1 else last_values
                advantage = rewards[step] - values[step] + discount_factor * dones[step].logical_not() * (next_values + lambda_parameter * advantage)
                returns[step].copy_(advantage + values[step])

            if normalize_returns:
                returns.copy_((returns - returns.mean()) / (returns.std() + 1e-8))

        # compute and normalize the advantages
        if advantages_dst is not None:
            advantages.copy_(returns - values)
            if normalize_advantages:
                advantages.copy_((advantages - advantages.mean()) / (advantages.std() + 1e-8))
        