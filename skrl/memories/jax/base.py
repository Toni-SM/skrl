from typing import List, Mapping, Optional, Tuple, Union

import csv
import datetime
import functools
import operator
import os
import gym
import gymnasium

import jax
import jax.numpy as jnp
import numpy as np

from skrl import config


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@jax.jit
def _copyto(dst, src):
    """NumPy function <function copyto at 0x7f804ee03430> not yet implemented
    """
    return dst.at[:].set(src)

@jax.jit
def _copyto_i(dst, src, i):
    return dst.at[i].set(src)

@jax.jit
def _copyto_i_j(dst, src, i, j):
    return dst.at[i, j].set(src)


class Memory:
    def __init__(self,
                 memory_size: int,
                 num_envs: int = 1,
                 device: Optional[jax.Device] = None,
                 export: bool = False,
                 export_format: str = "pt",  # TODO: set default format for jax
                 export_directory: str = "") -> None:
        """Base class representing a memory with circular buffers

        Buffers are jax or numpy arrays with shape (memory size, number of environments, data size).
        Circular buffers are implemented with two integers: a memory index and an environment index

        :param memory_size: Maximum number of elements in the first dimension of each internal storage
        :type memory_size: int
        :param num_envs: Number of parallel environments (default: ``1``)
        :type num_envs: int, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or jax.Device, optional
        :param export: Export the memory to a file (default: ``False``).
                       If True, the memory will be exported when the memory is filled
        :type export: bool, optional
        :param export_format: Export format (default: ``"pt"``).
                              Supported formats: torch (pt), numpy (np), comma separated values (csv)
        :type export_format: str, optional
        :param export_directory: Directory where the memory will be exported (default: ``""``).
                                 If empty, the agent's experiment directory will be used
        :type export_directory: str, optional

        :raises ValueError: The export format is not supported
        """
        self._jax = config.jax.backend == "jax"

        self.memory_size = memory_size
        self.num_envs = num_envs
        if device is None:
            self.device = jax.devices()[0]
        else:
            self.device = device if isinstance(device, jax.Device) else jax.devices(device)[0]

        # internal variables
        self.filled = False
        self.env_index = 0
        self.memory_index = 0

        self.tensors = {}
        self.tensors_view = {}
        self.tensors_keep_dimensions = {}
        self._views = True  # whether the views are not array copies

        self.sampling_indexes = None
        self.all_sequence_indexes = np.concatenate([np.arange(i, memory_size * num_envs + i, num_envs) for i in range(num_envs)])

        # exporting data
        self.export = export
        self.export_format = export_format
        self.export_directory = export_directory

        if not self.export_format in ["pt", "np", "csv"]:
            raise ValueError(f"Export format not supported ({self.export_format})")

    def __len__(self) -> int:
        """Compute and return the current (valid) size of the memory

        The valid size is calculated as the ``memory_size * num_envs`` if the memory is full (filled).
        Otherwise, the ``memory_index * num_envs + env_index`` is returned

        :return: Valid size
        :rtype: int
        """
        return self.memory_size * self.num_envs if self.filled else self.memory_index * self.num_envs + self.env_index

    def _get_space_size(self,
                        space: Union[int, Tuple[int], gym.Space, gymnasium.Space],
                        keep_dimensions: bool = False) -> Union[Tuple, int]:
        """Get the size (number of elements) of a space

        :param space: Space or shape from which to obtain the number of elements
        :type space: int, tuple or list of integers, gym.Space, or gymnasium.Space
        :param keep_dimensions: Whether or not to keep the space dimensions (default: ``False``)
        :type keep_dimensions: bool, optional

        :raises ValueError: If the space is not supported

        :return: Size of the space. If ``keep_dimensions`` is True, the space size will be a tuple
        :rtype: int or tuple of int
        """
        if type(space) in [int, float]:
            return (int(space),) if keep_dimensions else int(space)
        elif type(space) in [tuple, list]:
            return tuple(space) if keep_dimensions else np.prod(space)
        elif issubclass(type(space), gym.Space):
            if issubclass(type(space), gym.spaces.Discrete):
                return (1,) if keep_dimensions else 1
            elif issubclass(type(space), gym.spaces.Box):
                return tuple(space.shape) if keep_dimensions else np.prod(space.shape)
            elif issubclass(type(space), gym.spaces.Dict):
                if keep_dimensions:
                    raise ValueError("keep_dimensions=True cannot be used with Dict spaces")
                return sum([self._get_space_size(space.spaces[key]) for key in space.spaces])
        elif issubclass(type(space), gymnasium.Space):
            if issubclass(type(space), gymnasium.spaces.Discrete):
                return (1,) if keep_dimensions else 1
            elif issubclass(type(space), gymnasium.spaces.Box):
                return tuple(space.shape) if keep_dimensions else np.prod(space.shape)
            elif issubclass(type(space), gymnasium.spaces.Dict):
                if keep_dimensions:
                    raise ValueError("keep_dimensions=True cannot be used with Dict spaces")
                return sum([self._get_space_size(space.spaces[key]) for key in space.spaces])
        raise ValueError(f"Space type {type(space)} not supported")

    def _get_tensors_view(self, name):
        return self.tensors_view[name] if self._views else self.tensors[name].reshape(-1, self.tensors[name].shape[-1])

    def share_memory(self) -> None:
        """Share the tensors between processes
        """
        for tensor in self.tensors.values():
            pass

    def get_tensor_names(self) -> Tuple[str]:
        """Get the name of the internal tensors in alphabetical order

        :return: Tensor names without internal prefix (_tensor_)
        :rtype: tuple of strings
        """
        return sorted(self.tensors.keys())

    def get_tensor_by_name(self, name: str, keepdim: bool = True) -> Union[np.ndarray, jax.Array]:
        """Get a tensor by its name

        :param name: Name of the tensor to retrieve
        :type name: str
        :param keepdim: Keep the tensor's shape (memory size, number of environments, size) (default: ``True``)
                        If False, the returned tensor will have a shape of (memory size * number of environments, size)
        :type keepdim: bool, optional

        :raises KeyError: The tensor does not exist

        :return: Tensor
        :rtype: np.ndarray or jax.Array
        """
        return self.tensors[name] if keepdim else self._get_tensors_view(name)

    def set_tensor_by_name(self, name: str, tensor: Union[np.ndarray, jax.Array]) -> None:
        """Set a tensor by its name

        :param name: Name of the tensor to set
        :type name: str
        :param tensor: Tensor to set
        :type tensor: np.ndarray or jax.Array

        :raises KeyError: The tensor does not exist
        """
        if self._jax:
            self.tensors[name] = _copyto(self.tensors[name], tensor)
        else:
            np.copyto(self.tensors[name], tensor)

    def create_tensor(self,
                      name: str,
                      size: Union[int, Tuple[int], gym.Space, gymnasium.Space],
                      dtype: Optional[np.dtype] = None,
                      keep_dimensions: bool = False) -> bool:
        """Create a new internal tensor in memory

        The tensor will have a 3-components shape (memory size, number of environments, size).
        The internal representation will use _tensor_<name> as the name of the class property

        :param name: Tensor name (the name has to follow the python PEP 8 style)
        :type name: str
        :param size: Number of elements in the last dimension (effective data size).
                     The product of the elements will be computed for sequences or gym/gymnasium spaces
        :type size: int, tuple or list of integers or gym.Space
        :param dtype: Data type (np.dtype) (default: ``None``).
                      If None, the global default jax.numpy.float32 data type will be used
        :type dtype: np.dtype or None, optional
        :param keep_dimensions: Whether or not to keep the dimensions defined through the size parameter (default: ``False``)
        :type keep_dimensions: bool, optional

        :raises ValueError: The tensor name exists already but the size or dtype are different

        :return: True if the tensor was created, otherwise False
        :rtype: bool
        """
        # compute data size
        size = self._get_space_size(size, keep_dimensions)
        # check dtype and size if the tensor exists
        if name in self.tensors:
            tensor = self.tensors[name]
            if tensor.shape[-1] != size:
                raise ValueError(f"Size of tensor {name} ({size}) doesn't match the existing one ({tensor.shape[-1]})")
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError(f"Dtype of tensor {name} ({dtype}) doesn't match the existing one ({tensor.dtype})")
            return False
        # define tensor shape
        tensor_shape = (self.memory_size, self.num_envs, *size) if keep_dimensions else (self.memory_size, self.num_envs, size)
        view_shape = (-1, *size) if keep_dimensions else (-1, size)
        # create tensor (_tensor_<name>) and add it to the internal storage
        if self._jax:
            setattr(self, f"_tensor_{name}", jnp.zeros(tensor_shape, dtype=dtype))
        else:
            setattr(self, f"_tensor_{name}", np.zeros(tensor_shape, dtype=dtype))
        # update internal variables
        self.tensors[name] = getattr(self, f"_tensor_{name}")
        self.tensors_view[name] = self.tensors[name].reshape(*view_shape)
        self.tensors_keep_dimensions[name] = keep_dimensions
        # fill the tensors (float tensors) with NaN
        for name, tensor in self.tensors.items():
            if tensor.dtype == np.float32 or tensor.dtype == np.float64:
                if self._jax:
                    self.tensors[name] = _copyto(self.tensors[name], float("nan"))
                else:
                    self.tensors[name].fill(float("nan"))
        # check views
        if self._jax:
            self._views = False  # TODO: check if views are available
        else:
            self._views = self._views and self.tensors_view[name].base is self.tensors[name]
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

    def add_samples(self, **tensors: Mapping[str, Union[np.ndarray, jax.Array]]) -> None:
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
        tmp = tensors.get("states", tensors[next(iter(tensors))])  # ask for states first
        dim, shape = tmp.ndim, tmp.shape

        # multi environment (number of environments equals num_envs)
        if dim == 2 and shape[0] == self.num_envs:
            if self._jax:
                for name, tensor in tensors.items():
                    if name in self.tensors:
                        self.tensors[name] = _copyto_i(self.tensors[name], tensor, self.memory_index)
            else:
                for name, tensor in tensors.items():
                    if name in self.tensors:
                        self.tensors[name][self.memory_index] = tensor
            self.memory_index += 1
        # multi environment (number of environments less than num_envs)
        elif dim == 2 and shape[0] < self.num_envs:
            raise NotImplementedError  # TODO:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name] = self.tensors[name].at[self.memory_index, self.env_index:self.env_index + tensor.shape[0]].set(tensor)
            self.env_index += tensor.shape[0]
        # single environment - multi sample (number of environments greater than num_envs (num_envs = 1))
        elif dim == 2 and self.num_envs == 1:
            raise NotImplementedError  # TODO:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    num_samples = min(shape[0], self.memory_size - self.memory_index)
                    remaining_samples = shape[0] - num_samples
                    # copy the first n samples
                    self.tensors[name] = self.tensors[name].at[self.memory_index:self.memory_index + num_samples].set(tensor[:num_samples].unsqueeze(dim=1))
                    self.memory_index += num_samples
                    # storage remaining samples
                    if remaining_samples > 0:
                        self.tensors[name] = self.tensors[name].at[:remaining_samples].set(tensor[num_samples:].unsqueeze(dim=1))
                        self.memory_index = remaining_samples
        # single environment
        elif dim == 1:
            if self._jax:
                for name, tensor in tensors.items():
                    if name in self.tensors:
                        self.tensors[name] = _copyto_i_j(self.tensors[name], tensor, self.memory_index, self.env_index)
            else:
                for name, tensor in tensors.items():
                    if name in self.tensors:
                        self.tensors[name][self.memory_index, self.env_index] = tensor
            self.env_index += 1
        else:
            raise ValueError(f"Expected shape (number of environments = {self.num_envs}, data size), got {shape}")

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

    def sample(self,
               names: Tuple[str],
               batch_size: int,
               mini_batches: int = 1,
               sequence_length: int = 1) -> List[List[Union[np.ndarray, jax.Array]]]:
        """Data sampling method to be implemented by the inheriting classes

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param batch_size: Number of element to sample
        :type batch_size: int
        :param mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional
        :param sequence_length: Length of each sequence (default: ``1``)
        :type sequence_length: int, optional

        :raises NotImplementedError: The method has not been implemented

        :return: Sampled data from tensors sorted according to their position in the list of names.
                 The sampled tensors will have the following shape: (batch size, data size)
        :rtype: list of np.ndarray or jax.Array list
        """
        raise NotImplementedError("The sampling method (.sample()) is not implemented")

    def sample_by_index(self, names: Tuple[str], indexes: Union[tuple, np.ndarray, jax.Array], mini_batches: int = 1) -> List[List[Union[np.ndarray, jax.Array]]]:
        """Sample data from memory according to their indexes

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param indexes: Indexes used for sampling
        :type indexes: tuple or list, np.ndarray or jax.Array
        :param mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional

        :return: Sampled data from tensors sorted according to their position in the list of names.
                 The sampled tensors will have the following shape: (number of indexes, data size)
        :rtype: list of np.ndarray or jax.Array list
        """
        if mini_batches > 1:
            batches = np.array_split(indexes, mini_batches)
            views = [self._get_tensors_view(name) for name in names]
            return [[view[batch] for view in views] for batch in batches]
        return [[self._get_tensors_view(name)[indexes] for name in names]]

    def sample_all(self, names: Tuple[str], mini_batches: int = 1, sequence_length: int = 1) -> List[List[Union[np.ndarray, jax.Array]]]:
        """Sample all data from memory

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional
        :param sequence_length: Length of each sequence (default: ``1``)
        :type sequence_length: int, optional

        :return: Sampled data from memory.
                 The sampled tensors will have the following shape: (memory size * number of environments, data size)
        :rtype: list of np.ndarray or jax.Array list
        """
        # sequential order
        if sequence_length > 1:
            if mini_batches > 1:
                batches = np.array_split(self.all_sequence_indexes, len(self.all_sequence_indexes) // mini_batches)
                return [[self._get_tensors_view(name)[batch] for name in names] for batch in batches]
            return [[self._get_tensors_view(name)[self.all_sequence_indexes] for name in names]]

        # default order
        if mini_batches > 1:
            indexes = np.arange(self.memory_size * self.num_envs)
            batches = np.array_split(indexes, mini_batches)
            views = [self._get_tensors_view(name) for name in names]
            return [[view[batch] for view in views] for batch in batches]
        return [[self._get_tensors_view(name) for name in names]]

    def get_sampling_indexes(self) -> Union[tuple, np.ndarray, jax.Array]:
        """Get the last indexes used for sampling

        :return: Last sampling indexes
        :rtype: tuple or list, np.ndarray or jax.Array
        """
        return self.sampling_indexes

    def save(self, directory: str = "", format: str = "pt") -> None:
        """Save the memory to a file

        Supported formats:

        - PyTorch (pt)
        - NumPy (npz)
        - Comma-separated values (csv)

        :param directory: Path to the folder where the memory will be saved.
                          If not provided, the directory defined in the constructor will be used
        :type directory: str
        :param format: Format of the file where the memory will be saved (default: ``"pt"``)
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
            import torch
            torch.save({name: self.tensors[name] for name in self.get_tensor_names()}, memory_path)
        # numpy
        elif format == "npz":
            np.savez(memory_path, **{name: self.tensors[name].cpu().numpy() for name in self.get_tensor_names()})
        # comma-separated values
        elif format == "csv":
            # open csv writer # TODO: support keeping the dimensions
            with open(memory_path, "a") as file:
                writer = csv.writer(file)
                names = self.get_tensor_names()
                # write headers
                headers = [[f"{name}.{i}" for i in range(self.tensors[name].shape[-1])] for name in names]
                writer.writerow([item for sublist in headers for item in sublist])
                # write rows
                for i in range(len(self)):
                    writer.writerow(functools.reduce(operator.iconcat, [self.tensors[name].reshape(-1, self.tensors[name].shape[-1])[i].tolist() for name in names], []))
        # unsupported format
        else:
            raise ValueError(f"Unsupported format: {format}. Available formats: pt, csv, npz")

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
            import torch
            data = torch.load(path)
            for name in self.get_tensor_names():
                setattr(self, f"_tensor_{name}", jnp.array(data[name].cpu().numpy()))

        # numpy
        elif path.endswith(".npz"):
            data = np.load(path)
            for name in data:
                setattr(self, f"_tensor_{name}", jnp.array(data[name]))

        # comma-separated values
        elif path.endswith(".csv"):
            # TODO: load the memory from a csv
            pass

        # unsupported format
        else:
            raise ValueError(f"Unsupported format: {path}")
