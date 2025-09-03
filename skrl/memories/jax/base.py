from typing import List, Literal, Mapping, Optional, Sequence, Union

import csv
import datetime
import functools
import operator
import os
from abc import ABC, abstractmethod
import gymnasium

import jax
import jax.numpy as jnp
import numpy as np

from skrl import config
from skrl.utils.spaces.jax import compute_space_size


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@jax.jit
def _copyto(dst, src):
    """NumPy function <function copyto at 0x7f804ee03430> not yet implemented"""
    return dst.at[:].set(src)


@jax.jit
def _copyto_i(dst, src, i):
    return dst.at[i].set(src)


@jax.jit
def _copyto_i_j(dst, src, i, j):
    return dst.at[i, j].set(src)


class Memory(ABC):
    def __init__(
        self,
        *,
        memory_size: int,
        num_envs: int = 1,
        device: Optional[jax.Device] = None,
        export: bool = False,
        export_format: Literal["pt", "npz", "csv"] = "pt",
        export_directory: str = "",
    ) -> None:
        """Base class that represents a memory with circular buffers.

        Buffers are tensors with shape ``(memory_size, num_envs, data_size)``.
        Circular buffer is implemented with two integers: a memory index (``memory_index``, dimension 0)
        and an environment index (``env_index``, dimension 1).

        :param memory_size: Maximum number of elements in the first dimension for each tensor.
        :param num_envs: Number of parallel environments.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param export: Export the memory to a file. If ``True``, the memory will be exported once it is filled
            and before the circular buffer starts to overwrite the oldest data.
        :param export_format: File format to export the memory.
            Supported formats: PyTorch (``"pt"``), NumPy (``"npz"``) or comma separated values (``"csv"``).
        :param export_directory: Directory where the memory files will be exported.
            If not specified, the agent's experiment directory will be used.

        :raises ValueError: Unsupported export format.
        """
        self._jax = config.jax.backend == "jax"

        self.memory_size = memory_size
        self.num_envs = num_envs
        self.device = config.jax.parse_device(device)

        # internal variables
        self.filled = False
        # - indexes
        self.env_index = 0
        self.memory_index = 0
        # - allocation
        self.tensors = {}
        self.tensors_view = {}
        self._views = True  # whether the views are not array copies
        # - sampling
        self.sampling_indexes = None
        self.all_sequence_indexes = np.concatenate(
            [np.arange(i, memory_size * num_envs + i, num_envs) for i in range(num_envs)]
        )

        # exporting data
        self.export = export
        self.export_format = export_format
        self.export_directory = export_directory
        if self.export_format not in ["pt", "npz", "csv"]:
            raise ValueError(f"Unsupported export format: '{self.export_format}'")

    def __len__(self) -> int:
        """Compute and return the current (valid) size of the memory.

        The valid size is computed as:

        * ``memory_size * num_envs`` if the memory is full (filled)
        * ``memory_index * num_envs + env_index`` otherwise

        :return: Valid size.
        """
        return self.memory_size * self.num_envs if self.filled else self.memory_index * self.num_envs + self.env_index

    def _tensors_view(self, name: str) -> Union[np.ndarray, jax.Array]:
        return self.tensors_view[name] if self._views else self.tensors[name].reshape(-1, *self.tensors[name].shape[2:])

    def share_memory(self) -> None:
        """Set the tensors to be shared between processes."""
        for tensor in self.tensors.values():
            pass

    def get_tensor_names(self) -> Sequence[str]:
        """Get the name of the internal tensors, sorted alphabetically.

        :return: Tensor names without the internal prefix (``_tensor_``).
        """
        return sorted(self.tensors.keys())

    def get_tensor_by_name(self, name: str) -> Union[np.ndarray, jax.Array]:
        """Get a tensor by its name.

        :param name: Name of the tensor to get.

        :return: Tensor.

        :raises KeyError: The tensor does not exist.
        """
        return self.tensors[name]

    def set_tensor_by_name(self, name: str, tensor: Union[np.ndarray, jax.Array]) -> None:
        """Set a tensor by its name.

        :param name: Name of the tensor to set.
        :param tensor: Tensor to set.

        :raises KeyError: The tensor does not exist.
        """
        if self._jax:
            self.tensors[name] = _copyto(self.tensors[name], tensor)
        else:
            np.copyto(self.tensors[name], tensor)

    def create_tensor(
        self,
        name: str,
        *,
        size: Union[int, Sequence[int], gymnasium.Space, None],
        dtype: Optional[jnp.dtype] = None,
        keep_dimensions: bool = False,
    ) -> bool:
        """Create a new internal tensor in memory.

        The tensor will have a 3-dimensional with shape ``(memory_size, num_envs, data_size)``.
        The internal representation will use ``_tensor_<name>`` as the name of the class property.

        :param name: Tensor name (the name must follow the python PEP 8 style).
        :param size: Number of elements in the last dimension (effective data size).
            If a space is provided, the size will be computed as the number of elements occupied by the space.
        :param dtype: Data type. If not specified, the global default data type for PyTorch will be used.
        :param keep_dimensions: Whether to create a tensor with the original data dimensions.
            If enabled, only sequences of integers are supported as data ``size``.

        :return: True if the tensor was created, otherwise False.

        :raises ValueError: A tensor with the same name exists already but its size and/or dtype is different.
        """
        # don't create a tensor for None
        if size is None:
            return False
        if keep_dimensions:
            if not isinstance(size, (tuple, list)):
                raise ValueError("Only sequences of integers are supported as `size` when `keep_dimensions` is enabled")
        else:
            size = compute_space_size(size, occupied_size=True)
        # check dtype and size if the tensor exists already
        if name in self.tensors:
            tensor = self.tensors[name]
            if tensor.shape[-1] != size:
                raise ValueError(f"Tensor size ({size}) doesn't match the existing one ({tensor.shape[-1]}): '{name}'")
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError(f"Tensor dtype ({dtype}) doesn't match the existing one ({tensor.dtype}): '{name}'")
            return False
        # create tensor (_tensor_<name>) and add it to the internal storage
        shape = (self.memory_size, self.num_envs, *(size if keep_dimensions else [size]))
        if self._jax:
            setattr(self, f"_tensor_{name}", jnp.zeros(shape, device=self.device, dtype=dtype))
        else:
            setattr(self, f"_tensor_{name}", np.zeros(shape, dtype=dtype))
        # update internal variables
        self.tensors[name] = getattr(self, f"_tensor_{name}")
        self.tensors_view[name] = self.tensors[name].reshape((-1, *shape[2:]))
        # fill (float) tensors with NaN. This is useful for early misuse detection.
        for name, tensor in self.tensors.items():
            if tensor.dtype == np.float32 or tensor.dtype == np.float64:
                if self._jax:
                    with jax.default_device(self.device):
                        self.tensors[name] = _copyto(self.tensors[name], float("nan"))
                else:
                    tensor.fill(float("nan"))
        # check views
        if self._jax:
            self._views = False  # TODO: check if views are available
        else:
            self._views = self._views and self.tensors_view[name].base is self.tensors[name]
        return True

    def reset(self) -> None:
        """Reset the memory by clearing internal indexes and flags.

        .. note::

            Old data will be retained until overwritten, but access through the available methods will not be guaranteed.

        Default values of the internal indexes and flags after the reset:

        * ``filled``: ``False``
        * ``env_index``: 0
        * ``memory_index``: 0
        """
        self.filled = False
        self.env_index = 0
        self.memory_index = 0

    def add_samples(self, **tensors: Mapping[str, Union[np.ndarray, jax.Array]]) -> None:
        """Add/store samples in memory.

        .. important::

            All tensors must have the same dimensions (2 dimensions) and shape: ``(current_num_envs, data_size)``.
            If the tensors have one dimension, it is assumed that ``current_num_envs`` is 1.

            No check is performed for compatibility of the shapes or for memory write overflow.

        According to the number of environments, the following behavior is performed:

        * ``current_num_envs = num_envs``: store samples and increment the memory index (1st index) by one.
        * ``current_num_envs < num_envs``: store samples and increment the environment index (2nd index)
          by the current number of environments.
        * ``current_num_envs > num_envs`` and ``num_envs = 1``: store multiple samples and increment the memory index
          (1st index) by the number of samples. If the number of samples is greater than the remaining memory size,
          the memory will be filled and circular buffer will overwrite the oldest data with the remaining samples.

        :param tensors: Sample data, as key-value arguments (keys: tensor names). Non-existing tensors will be skipped.

        :raises ValueError: No tensors were provided or the tensors have incompatible shapes.
        """
        if not tensors:
            raise ValueError(
                "There are no samples. Provide samples as key-value arguments, where keys are the tensor names"
            )

        # dimensions and shapes of the tensors (assume all tensors have the dimensions of the first tensor)
        tmp = tensors.get("observations", tensors[next(iter(tensors))])  # ask for observations first
        dim, shape = tmp.ndim, tmp.shape

        # multi environment (current_num_envs = num_envs)
        if dim == 2 and shape[0] == self.num_envs:
            if self._jax:
                for name, tensor in tensors.items():
                    if name in self.tensors and tensor is not None:
                        self.tensors[name] = _copyto_i(self.tensors[name], tensor, self.memory_index)
            else:
                for name, tensor in tensors.items():
                    if name in self.tensors and tensor is not None:
                        self.tensors[name][self.memory_index] = tensor
            self.memory_index += 1
        # multi environment (current_num_envs < num_envs)
        elif dim == 2 and shape[0] < self.num_envs:
            raise NotImplementedError  # TODO: implement
        # single environment - multi sample (num_envs = 1, current_num_envs > 1)
        elif dim == 2 and self.num_envs == 1:
            raise NotImplementedError  # TODO: implement
        # single environment (current_num_envs = 1, implicit)
        elif dim == 1:
            if self._jax:
                for name, tensor in tensors.items():
                    if name in self.tensors and tensor is not None:
                        self.tensors[name] = _copyto_i_j(self.tensors[name], tensor, self.memory_index, self.env_index)
            else:
                for name, tensor in tensors.items():
                    if name in self.tensors and tensor is not None:
                        self.tensors[name][self.memory_index, self.env_index] = tensor
            self.env_index += 1
        else:
            raise ValueError(
                f"Expected shape (current_num_envs, data_size) where current_num_envs <= {self.num_envs}, got {shape}"
            )

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

    @abstractmethod
    def sample(
        self, names: Sequence[str], *, batch_size: int, mini_batches: int = 1, sequence_length: int = 1
    ) -> List[List[Union[np.ndarray, jax.Array]]]:
        """Data sampling method to be implemented by the inheriting classes.

        :param names: Tensors names from which to obtain the samples.
        :param batch_size: Number of elements to sample.
        :param mini_batches: Number of mini-batches to sample.
        :param sequence_length: Length of each sequence.

        :return: Sampled data from tensors sorted according to their position in the list of names.
            The sampled tensors will have the following shape: ``(batch_size, data_size)``.
        """
        pass

    def sample_by_index(
        self, names: Sequence[str], *, indexes: Union[tuple, np.ndarray, jax.Array], mini_batches: int = 1
    ) -> List[List[Union[np.ndarray, jax.Array]]]:
        """Sample data from memory according to their indexes.

        :param names: Tensors names from which to obtain the samples.
        :param indexes: Indexes used for sampling.
        :param mini_batches: Number of mini-batches to sample.

        :return: Sampled data from tensors sorted according to their position in the list of names.
            The sampled tensors will have the following shape: ``(number_of_indexes, data_size)``.
        """
        if mini_batches > 1:
            batches = np.array_split(indexes, mini_batches)
            views = [self._tensors_view(name) if name in self.tensors else None for name in names]
            return [[None if view is None else view[batch] for view in views] for batch in batches]
        return [[self._tensors_view(name)[indexes] if name in self.tensors else None for name in names]]

    def sample_all(
        self, names: Sequence[str], *, mini_batches: int = 1, sequence_length: int = 1
    ) -> List[List[Union[np.ndarray, jax.Array]]]:
        """Sample all data from memory.

        :param names: Tensors names from which to obtain the samples.
        :param mini_batches: Number of mini-batches to sample.
        :param sequence_length: Length of each sequence.

        :return: Sampled data from memory.
            The sampled tensors will have the following shape: ``(memory_size * number_of_environments, data_size)``.
        """
        # sequential order
        if sequence_length > 1:
            if mini_batches > 1:
                batches = np.array_split(self.all_sequence_indexes, mini_batches)
                views = [self._tensors_view(name) if name in self.tensors else None for name in names]
                return [[None if view is None else view[batch] for view in views] for batch in batches]
            return [
                [
                    self._tensors_view(name)[self.all_sequence_indexes] if name in self.tensors else None
                    for name in names
                ]
            ]
        # default order
        if mini_batches > 1:
            batch_size = (self.memory_size * self.num_envs) // mini_batches
            batches = [(batch_size * i, batch_size * (i + 1)) for i in range(mini_batches)]
            views = [self._tensors_view(name) if name in self.tensors else None for name in names]
            return [[None if view is None else view[batch[0] : batch[1]] for view in views] for batch in batches]
        return [[self._tensors_view(name) if name in self.tensors else None for name in names]]

    def get_sampling_indexes(self) -> Union[tuple, np.ndarray, jax.Array]:
        """Get the last indexes used for sampling.

        :return: Last sampling indexes.
        """
        return self.sampling_indexes

    def save(self, directory: str = "", *, format: Literal["pt", "npz", "csv"] = "pt") -> None:
        """Save the memory to a file.

        :param directory: Path to the folder where the memory will be saved.
            If not provided, the directory defined in the constructor will be used.
        :param format: Format of the file where the memory will be saved.
            Supported formats: PyTorch (``"pt"``), NumPy (``"npz"``) or comma separated values (``"csv"``).

        :raises ValueError: Unsupported format.
        """
        raise NotImplementedError("Saving is not supported yet")

    def load(self, path: str) -> None:
        """Load the memory from a file.

        Supported formats: PyTorch (``"pt"``), NumPy (``"npz"``) or comma separated values (``"csv"``).

        :param path: Path to the file where the memory will be loaded.

        :raises ValueError: Unsupported format.
        """
        raise NotImplementedError("Loading is not supported yet")
