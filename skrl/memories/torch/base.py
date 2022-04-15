from typing import Union, Tuple, List

import os
import csv
import gym
import operator
import datetime
import functools
import numpy as np

import torch
from torch.utils.data.sampler import BatchSampler


class Memory:
    def __init__(self, 
                 memory_size: int, 
                 num_envs: int = 1, 
                 device: Union[str, torch.device] = "cuda:0", 
                 export: bool = False, 
                 export_format: str = "pt", 
                 export_directory: str = "") -> None:
        """Base class representing a memory with circular buffers

        Buffers are torch tensors with shape (memory size, number of environments, data size).
        Circular buffers are implemented with two integers: a memory index and an environment index
        
        :param memory_size: Maximum number of elements in the first dimension of each internal storage
        :type memory_size: int
        :param num_envs: Number of parallel environments (default: 1)
        :type num_envs: int, optional
        :param device: Device on which a torch tensor is or will be allocated (default: "cuda:0")
        :type device: str or torch.device, optional
        :param export: Export the memory to a file (default: False).
                       If True, the memory will be exported when the memory is filled
        :type export: bool, optional
        :param export_format: Export format (default: "pt").
                              Supported formats: torch (pt), numpy (np), comma separated values (csv)
        :type export_format: str, optional
        :param export_directory: Directory where the memory will be exported (default: "").
                                 If empty, the agent's experiment directory will be used
        :type export_directory: str, optional

        :raises ValueError: The export format is not supported
        """
        self.memory_size = memory_size
        self.num_envs = num_envs
        self.device = torch.device(device)

        # internal variables
        self.filled = False
        self.env_index = 0
        self.memory_index = 0

        self.tensors = {}
        self.tensors_view = {}

        # exporting data
        self.export = export
        self.export_format = export_format
        self.export_directory = export_directory

        if not self.export_format in ["pt", "np", "csv"]:
            raise ValueError("Export format not supported ({})".format(self.export_format))

    def __len__(self) -> int:
        """Compute and return the current (valid) size of the memory
        
        The valid size is calculated as the ``memory_size * num_envs`` if the memory is full (filled).
        Otherwise, the ``memory_index * num_envs + env_index`` is returned

        :return: Valid size
        :rtype: int
        """
        return self.memory_size * self.num_envs if self.filled else self.memory_index * self.num_envs + self.env_index
        
    def _get_space_size(self, space: Union[int, Tuple[int], gym.Space]) -> int:
        """Get the size (number of elements) of a space

        :param space: Space or shape from which to obtain the number of elements
        :type space: int, tuple or list of integers, or gym.Space

        :raises ValueError: If the space is not supported

        :return: Size of the space data
        :rtype: Space size (number of elements)
        """
        if type(space) in [int, float]:
            return int(space)
        elif type(space) in [tuple, list]:
            return np.prod(space)
        elif issubclass(type(space), gym.Space):
            if issubclass(type(space), gym.spaces.Discrete):
                return 1
            elif issubclass(type(space), gym.spaces.Box):
                return np.prod(space.shape)
            elif issubclass(type(space), gym.spaces.Dict):
                return sum([self._get_space_size(space.spaces[key]) for key in space.spaces])
        raise ValueError("Space type {} not supported".format(type(space)))

    def share_memory(self) -> None:
        """Share the tensors between processes
        """
        for tensor in self.tensors.values():
            tensor.share_memory_()

    def get_tensor_names(self) -> Tuple[str]:
        """Get the name of the internal tensors in alphabetical order

        :return: Tensor names without internal prefix (_tensor_)
        :rtype: tuple of strings
        """
        return sorted(self.tensors.keys())

    def create_tensor(self, name: str, size: Union[int, Tuple[int], gym.Space], dtype: Union[torch.dtype, None] = None) -> bool:
        """Create a new internal tensor in memory
        
        The tensor will have a 3-components shape (memory size, number of environments, size).
        The internal representation will use _tensor_<name> as the name of the class property

        :param name: Tensor name (the name has to follow the python PEP 8 style)
        :type name: str
        :param size: Number of elements in the last dimension (effective data size).
                     The product of the elements will be computed for collections or gym spaces types
        :type size: int, tuple or list of integers or gym.Space
        :param dtype: Data type (torch.dtype).
                      If None, the global default torch data type will be used (default)
        :type dtype: torch.dtype or None, optional
        
        :raises ValueError: The tensor name exists already but the size or dtype are different

        :return: True if the tensor was created, otherwise False
        :rtype: bool
        """
        # compute data size
        size = self._get_space_size(size)
        # check dtype and size if the tensor exists
        if name in self.tensors:
            tensor = self.tensors[name]
            if tensor.size(-1) != size:
                raise ValueError("The size of the tensor {} ({}) doesn't match the existing one ({})".format(name, size, tensor.size(-1)))
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError("The dtype of the tensor {} ({}) doesn't match the existing one ({})".format(name, dtype, tensor.dtype))
            return False
        # create tensor (_tensor_<name>) and add it to the internal storage
        setattr(self, "_tensor_{}".format(name), torch.zeros((self.memory_size, self.num_envs, size), device=self.device, dtype=dtype))
        self.tensors[name] = getattr(self, "_tensor_{}".format(name))
        self.tensors_view[name] = self.tensors[name].view(-1, self.tensors[name].size(-1))
        # fill the tensors (float tensors) with NaN
        for tensor in self.tensors.values():
            if torch.is_floating_point(tensor):
                tensor.fill_(float("nan"))
        return True

    def reset(self) -> None:
        """Reset the memory by cleaning internal indexes and flags

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
        """Record samples in memory

        Samples should be a tensor with 2-components shape (number of environments, data size).
        All tensors must be of the same shape

        According to the number of environments, the following classification is made:

        - one environment:
          Store a single sample (tensors with one dimension) and increment the environment index (second index) by one

        - number of environments less than num_envs:
          Store the samples and increment the environment index (second index) by the number of the environments
        
        - number of environments equals num_envs:
          Store the samples and increment the memory index (first index) by one

        :param tensors: Sampled data as key-value arguments where the keys are the names of the tensors to be modified.
                        Non-existing tensors will be skipped
        :type tensors: dict

        :raises ValueError: No tensors were provided or the tensors have incompatible shapes
        """
        if not tensors:
            raise ValueError("No samples to be recorded in memory. Pass samples as key-value arguments (where key is the tensor name)")

        # dimensions and shapes of the tensors (assume all tensors have the dimensions of the first tensor)
        tmp = tensors[next(iter(tensors))]
        dim, shape = tmp.ndim, tmp.shape

        # multi environment (number of environments less than num_envs)
        if dim == 2 and shape[0] < self.num_envs:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name][self.memory_index, self.env_index:self.env_index + tensor.shape[0]].copy_(tensor)
            self.env_index += tensor.shape[0]
        # multi environment (number of environments equals num_envs)
        elif dim == 2 and shape[0] == self.num_envs:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name][self.memory_index].copy_(tensor)
            self.memory_index += 1
        # single environment
        elif dim == 1:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name][self.memory_index, self.env_index].copy_(tensor)
            self.env_index += 1
        else:
            raise ValueError("Expected tensors with 2-components shape (number of environments, data size), got {}".format(shape))

        # update indexes and flags
        if self.env_index >= self.num_envs:
            self.env_index = 0
            self.memory_index += 1
        if self.memory_index >= self.memory_size:
            self.memory_index = 0
            self.filled = True

            # export tensors to file
            if self.export:
                self.save(directory=self.export_directory, format=self.export_format)

    def sample(self, names: Tuple[str], batch_size: int, mini_batches: int = 1) -> List[List[torch.Tensor]]:
        """Data sampling method to be implemented by the inheriting classes

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param batch_size: Number of element to sample
        :type batch_size: int
        :param mini_batches: Number of mini-batches to sample (default: 1)
        :type mini_batches: int, optional
        
        :raises NotImplementedError: The method has not been implemented
        
        :return: Sampled data from tensors sorted according to their position in the list of names.
                 The sampled tensors will have the following shape: (batch size, data size)
        :rtype: list of torch.Tensor list
        """
        raise NotImplementedError("The sampling method (.sample()) is not implemented")

    def sample_by_index(self, names: Tuple[str], indexes: Union[tuple, np.ndarray, torch.Tensor], mini_batches: int = 1) -> List[List[torch.Tensor]]:
        """Sample data from memory according to their indexes

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param indexes: Indexes used for sampling
        :type indexes: tuple or list, numpy.ndarray or torch.Tensor
        :param mini_batches: Number of mini-batches to sample (default: 1)
        :type mini_batches: int, optional

        :return: Sampled data from tensors sorted according to their position in the list of names.
                 The sampled tensors will have the following shape: (number of indexes, data size)
        :rtype: list of torch.Tensor list
        """
        if mini_batches > 1:
            batches = BatchSampler(indexes, batch_size=len(indexes) // mini_batches, drop_last=True)
            return [[self.tensors_view[name][batch] for name in names] for batch in batches]
        return [[self.tensors_view[name][indexes] for name in names]]

    def sample_all(self, names: Tuple[str], mini_batches: int = 1) -> List[List[torch.Tensor]]:
        """Sample all data from memory
        
        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param mini_batches: Number of mini-batches to sample (default: 1)
        :type mini_batches: int, optional

        :return: Sampled data from memory.
                 The sampled tensors will have the following shape: (memory size * number of environments, data size)
        :rtype: list of torch.Tensor list
        """
        if mini_batches > 1:
            indexes = np.arange(self.memory_size * self.num_envs)
            batches = BatchSampler(indexes, batch_size=len(indexes) // mini_batches, drop_last=True)
            return [[self.tensors_view[name][batch] for name in names] for batch in batches]
        return [[self.tensors_view[name] for name in names]]

    def compute_functions(self, states_src: str = "states", actions_src: str = "actions", rewards_src: str = "rewards", next_states_src: str = "next_states", dones_src: str = "dones", values_src: str = "values", returns_dst: Union[str, None] = None, advantages_dst: Union[str, None] = None, last_values: Union[torch.Tensor, None] = None, hyperparameters: dict = {"discount_factor": 0.99, "lambda_coefficient": 0.95, "normalize_returns": False, "normalize_advantages": True}) -> None:
        """Compute the following functions for the given tensor names

        Available functions:
        - Returns (total discounted reward)
        - Advantages (total discounted reward - baseline)

        :param states_src: Name of the tensor containing the states (default: "states")
        :type states_src: str, optional
        :param actions_src: Name of the tensor containing the actions (default: "actions")
        :type actions_src: str, optional
        :param rewards_src: Name of the tensor containing the rewards (default: "rewards")
        :type rewards_src: str, optional
        :param next_states_src: Name of the tensor containing the next states (default: "next_states")
        :type next_states_src: str, optional
        :param dones_src: Name of the tensor containing the dones (default: "dones")
        :type dones_src: str, optional
        :param values_src: Name of the tensor containing the values (default: "values")
        :type values_src: str, optional
        :param returns_dst: Name of the tensor where the returns will be stored (default: None)
        :type returns_dst: str or None, optional
        :param advantages_dst: Name of the tensor where the advantages will be stored (default: None)
        :type advantages_dst: str or None, optional
        :param last_values: Last values (default: None).
                            If None, the last values will be obtained from the tensor containing the values
        :type last_values: torch.Tensor or None, optional
        :param hyperparameters: Hyperparameters to control the computation of the functions.
                                The following hyperparameters are expected:

                                  * **discount_factor** (float) - Discount factor (gamma) for computing returns and advantages (default: 0.99)
                                
                                  * **lambda_coefficient** (float) - TD(lambda) coefficient (lam) for computing returns and advantages (default: 0.95)
                                
                                  * **normalize_returns** (bool) - If True, the returns will be normalized (default: False)
                                
                                  * **normalize_advantages** (bool) - If True, the advantages will be normalized (default: True)
        :type hyperparameters: dict, optional
        """
        # TODO: compute functions attending the circular buffer logic
        # TODO: get last values from the last samples (if not provided) and ignore them in the computation

        # get source and destination tensors
        rewards = self.tensors[rewards_src]
        dones = self.tensors[dones_src]
        values = self.tensors[values_src]
        
        returns = self.tensors[returns_dst] if returns_dst is not None else torch.zeros_like(rewards)
        advantages = self.tensors[advantages_dst] if advantages_dst is not None else torch.zeros_like(rewards)

        # hyperarameters
        discount_factor = hyperparameters.get("discount_factor", 0.99)
        lambda_coefficient = hyperparameters.get("lambda_coefficient", 0.95)
        normalize_returns = hyperparameters.get("normalize_returns", False)
        normalize_advantages = hyperparameters.get("normalize_advantages", True)

        # compute and normalize the returns
        if returns_dst is not None or advantages_dst is not None:
            advantage = 0
            for step in reversed(range(self.memory_size)):
                next_values = values[step + 1] if step < self.memory_size - 1 else last_values
                advantage = rewards[step] - values[step] + discount_factor * dones[step].logical_not() * (next_values + lambda_coefficient * advantage)
                returns[step].copy_(advantage + values[step])

            if normalize_returns:
                returns.copy_((returns - returns.mean()) / (returns.std() + 1e-8))

        # compute and normalize the advantages
        if advantages_dst is not None:
            advantages.copy_(returns - values)
            if normalize_advantages:
                advantages.copy_((advantages - advantages.mean()) / (advantages.std() + 1e-8))
        
    def save(self, directory: str = "", format: str = "pt") -> None:
        """Save the memory to a file

        Supported formats:
        
        - PyTorch (pt)
        - NumPy (npz)
        - Comma-separated values (csv)

        :param directory: Path to the folder where the memory will be saved.
                          If not provided, the directory defined in the constructor will be used
        :type directory: str
        :param format: Format of the file where the memory will be saved (default: "pt")
        :type format: str, optional

        :raises ValueError: If the format is not supported
        """
        if not directory:
            directory = self.export_directory
        os.makedirs(os.path.join(directory, "memories"), exist_ok=True)
        memory_path = os.path.join(directory, "memories", \
            "{}_memory_{}.{}".format(datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"), hex(id(self)), format))
        
        # torch
        if format == "pt":
            torch.save({name: self.tensors[name] for name in self.get_tensor_names()}, memory_path)
        # numpy
        elif format == "npz":
            np.savez(memory_path, **{name: self.tensors[name].cpu().numpy() for name in self.get_tensor_names()})
        # comma-separated values
        elif format == "csv":
            # open csv writer
            with open(memory_path, "a") as file:
                writer = csv.writer(file)
                names = self.get_tensor_names()
                # write headers
                headers = [["{}.{}".format(name, i) for i in range(self.tensors_view[name].shape[-1])] for name in names]
                writer.writerow([item for sublist in headers for item in sublist])
                # write rows
                for i in range(len(self)):
                    writer.writerow(functools.reduce(operator.iconcat, [self.tensors_view[name][i].tolist() for name in names], []))
        # unsupported format
        else:
            raise ValueError("Unsupported format: {}. Available formats: pt, csv, npz".format(format))

    def load(self, path: str) -> None:
        """Load the memory from a file

        Supported formats:
        - PyTorch (pt)
        - NumPy (npz)
        - Comma-separated values (csv)

        :param path: Path to the file where the memory will be loaded
        :type path: str

        :raises ValueError: If the format is not supported
        """
        # torch
        if path.endswith(".pt"):
            data = torch.load(path)
            for name in self.get_tensor_names():
                setattr(self, "_tensor_{}".format(name), data[name])
        
        # numpy
        elif path.endswith(".npz"):
            data = np.load(path)
            for name in data:
                setattr(self, "_tensor_{}".format(name), torch.tensor(data[name]))
        
        # comma-separated values
        elif path.endswith(".csv"):
            # TODO: load the memory from a csv
            pass
        
        # unsupported format
        else:
            raise ValueError("Unsupported format: {}".format(path))
