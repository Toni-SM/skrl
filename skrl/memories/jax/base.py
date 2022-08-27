from typing import Optional, Union, Mapping, Tuple, List

import os
import csv
import gym
import operator
import datetime
import warnings
import functools
import numpy as np

import jax
import jaxlib
import jax.numpy as jnp


class Memory:
    def __init__(self, 
                 memory_size: int, 
                 num_envs: int = 1, 
                 device: Optional[jaxlib.xla_extension.Device] = None, 
                 export: bool = False, 
                 export_format: str = "pt",  # TODO: set default format for jax
                 export_directory: str = "") -> None:
        """Base class representing a memory with circular buffers

        Buffers are jax arrays with shape (memory size, number of environments, data size).
        Circular buffers are implemented with two integers: a memory index and an environment index
        
        :param memory_size: Maximum number of elements in the first dimension of each internal storage
        :type memory_size: int
        :param num_envs: Number of parallel environments (default: 1)
        :type num_envs: int, optional
        :param device: Device on which an array is or will be allocated (default: None)
        :type device: jaxlib.xla_extension.Device, optional
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
        self.device = jax.devices()[0] if device is None else device

        # internal variables
        self.filled = False
        self.env_index = 0
        self.memory_index = 0

        self.tensors = {}
        # TODO: add views in future implementations if possible with jax

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
        warnings.warn("Memory sharing not implemented for jax backend")
        pass

    def get_tensor_names(self) -> Tuple[str]:
        """Get the name of the internal tensors in alphabetical order

        :return: Tensor names without internal prefix (_tensor_)
        :rtype: tuple of strings
        """
        return sorted(self.tensors.keys())

    def get_tensor_by_name(self, name: str, keepdim: bool = True) -> jaxlib.xla_extension.DeviceArray:
        """Get a tensor by its name

        :param name: Name of the tensor to retrieve
        :type name: str
        :param keepdim: Keep the tensor's shape (memory size, number of environments, size) (default: True)
                        If False, the returned tensor will have a shape of (memory size * number of environments, size)
        :type keepdim: bool, optional

        :raises KeyError: The tensor does not exist

        :return: Tensor
        :rtype: jaxlib.xla_extension.DeviceArray
        """
        return self.tensors[name] if keepdim else self.tensors[name].reshape(-1, self.tensors[name].shape[-1])

    def set_tensor_by_name(self, name: str, tensor: jaxlib.xla_extension.DeviceArray) -> None:
        """Set a tensor by its name

        :param name: Name of the tensor to set
        :type name: str
        :param tensor: Tensor to set
        :type tensor: jaxlib.xla_extension.DeviceArray

        :raises KeyError: The tensor does not exist
        """
        # TODO: Numpy function <function copyto at 0x7f804ee03430> not yet implemented  
        self.tensors[name] = self.tensors[name].at[:].set(tensor)

    def create_tensor(self, 
                      name: str, 
                      size: Union[int, Tuple[int], gym.Space], 
                      dtype: Optional[np.dtype] = None) -> bool:
        """Create a new internal tensor in memory
        
        The tensor will have a 3-components shape (memory size, number of environments, size).
        The internal representation will use _tensor_<name> as the name of the class property

        :param name: Tensor name (the name has to follow the python PEP 8 style)
        :type name: str
        :param size: Number of elements in the last dimension (effective data size).
                     The product of the elements will be computed for collections or gym spaces types
        :type size: int, tuple or list of integers or gym.Space
        :param dtype: Data type (np.dtype).
                      If None, the global default jax.numpy.float32 data type will be used (default)
        :type dtype: np.dtype, optional
        
        :raises ValueError: The tensor name exists already but the size or dtype are different

        :return: True if the tensor was created, otherwise False
        :rtype: bool
        """
        # compute data size
        size = self._get_space_size(size)
        dtype = np.float32 if dtype is None else dtype
        # check dtype and size if the tensor exists
        if name in self.tensors:
            tensor = self.tensors[name]
            if tensor.shape[-1] != size:
                raise ValueError("The size of the tensor {} ({}) doesn't match the existing one ({})".format(name, size, tensor.shape[-1]))
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError("The dtype of the tensor {} ({}) doesn't match the existing one ({})".format(name, dtype, tensor.dtype))
            return False
        # create tensor (_tensor_<name>) and add it to the internal storage
        setattr(self, "_tensor_{}".format(name), jnp.zeros((self.memory_size, self.num_envs, size), dtype=dtype))
        self.tensors[name] = getattr(self, "_tensor_{}".format(name))
        # fill the tensors (float tensors) with NaN  
        # TODO: AttributeError: 'DeviceArray' object has no attribute 'fill'
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

    def add_samples(self, **tensors: Mapping[str, jaxlib.xla_extension.DeviceArray]) -> None:
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
        :type tensors: Mapping[str, jaxlib.xla_extension.DeviceArray]

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
                    self.tensors[name] = self.tensors[name].at[self.memory_index, self.env_index:self.env_index + tensor.shape[0]].set(tensor)
            self.env_index += tensor.shape[0]
        # multi environment (number of environments equals num_envs)
        elif dim == 2 and shape[0] == self.num_envs:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name] = self.tensors[name].at[self.memory_index].set(tensor)
            self.memory_index += 1
        # single environment - multi sample (number of environments greater than num_envs (num_envs = 1))
        elif dim == 2 and self.num_envs == 1:
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
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name] = self.tensors[name][self.memory_index, self.env_index].set(tensor)
            self.env_index += 1
        else:
            raise ValueError("Expected tensors with 2-components shape (number of environments = {}, data size), got {}".format(self.num_envs, shape))

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

    def sample(self, names: Tuple[str], batch_size: int, mini_batches: int = 1) -> List[List[jaxlib.xla_extension.DeviceArray]]:
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
        :rtype: list of jaxlib.xla_extension.DeviceArray list
        """
        raise NotImplementedError("The sampling method (.sample()) is not implemented")

    def sample_by_index(self, names: Tuple[str], indexes: Union[tuple, np.ndarray, jaxlib.xla_extension.DeviceArray], mini_batches: int = 1) -> List[List[jaxlib.xla_extension.DeviceArray]]:
        """Sample data from memory according to their indexes

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param indexes: Indexes used for sampling
        :type indexes: tuple or list, numpy.ndarray or jaxlib.xla_extension.DeviceArray
        :param mini_batches: Number of mini-batches to sample (default: 1)
        :type mini_batches: int, optional

        :return: Sampled data from tensors sorted according to their position in the list of names.
                 The sampled tensors will have the following shape: (number of indexes, data size)
        :rtype: list of jaxlib.xla_extension.DeviceArray list
        """
        if mini_batches > 1:
            batches = jnp.array_split(indexes, mini_batches)  # FIXME: convert tuple or list to array
            return [[self.tensors[name].reshape(-1, self.tensors[name].shape[-1])[batch] for name in names] for batch in batches]
        return [[self.tensors[name].reshape(-1, self.tensors[name].shape[-1])[indexes] for name in names]]

    def sample_all(self, names: Tuple[str], mini_batches: int = 1) -> List[List[jaxlib.xla_extension.DeviceArray]]:
        """Sample all data from memory
        
        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param mini_batches: Number of mini-batches to sample (default: 1)
        :type mini_batches: int, optional

        :return: Sampled data from memory.
                 The sampled tensors will have the following shape: (memory size * number of environments, data size)
        :rtype: list of jaxlib.xla_extension.DeviceArray list
        """
        if mini_batches > 1:
            batches = jnp.array_split(np.arange(self.memory_size * self.num_envs), mini_batches)
            return [[self.tensors[name].reshape(-1, self.tensors[name].shape[-1])[batch] for name in names] for batch in batches]
        return [[self.tensors[name].reshape(-1, self.tensors[name].shape[-1]) for name in names]]
        
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
            import torch
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
                headers = [["{}.{}".format(name, i) for i in range(self.tensors[name].shape[-1])] for name in names]
                writer.writerow([item for sublist in headers for item in sublist])
                # write rows
                for i in range(len(self)):
                    writer.writerow(functools.reduce(operator.iconcat, [self.tensors[name].reshape(-1, self.tensors[name].shape[-1])[i].tolist() for name in names], []))
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
            import torch
            data = torch.load(path)
            for name in self.get_tensor_names():
                setattr(self, "_tensor_{}".format(name), jnp.array(data[name].cpu().numpy()))
        
        # numpy
        elif path.endswith(".npz"):
            data = np.load(path)
            for name in data:
                setattr(self, "_tensor_{}".format(name), jnp.array(data[name]))
        
        # comma-separated values
        elif path.endswith(".csv"):
            # TODO: load the memory from a csv
            pass
        
        # unsupported format
        else:
            raise ValueError("Unsupported format: {}".format(path))
