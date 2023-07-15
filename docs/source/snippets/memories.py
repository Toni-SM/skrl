# [start-base-class-torch]
from typing import Union, Tuple, List

import torch

from skrl.memories.torch import Memory


class CustomMemory(Memory):
    def __init__(self, memory_size: int, num_envs: int = 1, device: Union[str, torch.device] = "cuda:0") -> None:
        """Custom memory

        :param memory_size: Maximum number of elements in the first dimension of each internal storage
        :type memory_size: int
        :param num_envs: Number of parallel environments (default: 1)
        :type num_envs: int, optional
        :param device: Device on which a torch tensor is or will be allocated (default: "cuda:0")
        :type device: str or torch.device, optional
        """
        super().__init__(memory_size, num_envs, device)

    def sample(self, names: Tuple[str], batch_size: int, mini_batches: int = 1) -> List[List[torch.Tensor]]:
        """Sample a batch from memory

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param batch_size: Number of element to sample
        :type batch_size: int
        :param mini_batches: Number of mini-batches to sample (default: 1)
        :type mini_batches: int, optional

        :return: Sampled data from tensors sorted according to their position in the list of names.
                 The sampled tensors will have the following shape: (batch size, data size)
        :rtype: list of torch.Tensor list
        """
        # ================================
        # - sample a batch from memory.
        #   It is possible to generate only the sampling indexes and call self.sample_by_index(...)
        # ================================
# [end-base-class-torch]


# [start-base-class-jax]
from typing import Optional, Union, Tuple, List

import jaxlib
import jax.numpy as jnp

from skrl.memories.jax import Memory


class CustomMemory(Memory):
    def __init__(self, memory_size: int,
                 num_envs: int = 1,
                 device: Optional[jaxlib.xla_extension.Device] = None) -> None:
        """Custom memory

        :param memory_size: Maximum number of elements in the first dimension of each internal storage
        :type memory_size: int
        :param num_envs: Number of parallel environments (default: 1)
        :type num_envs: int, optional
        :param device: Device on which an array is or will be allocated (default: None)
        :type device: jaxlib.xla_extension.Device, optional
        """
        super().__init__(memory_size, num_envs, device)

    def sample(self, names: Tuple[str], batch_size: int, mini_batches: int = 1) -> List[List[jnp.ndarray]]:
        """Sample a batch from memory

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param batch_size: Number of element to sample
        :type batch_size: int
        :param mini_batches: Number of mini-batches to sample (default: 1)
        :type mini_batches: int, optional

        :return: Sampled data from tensors sorted according to their position in the list of names.
                 The sampled tensors will have the following shape: (batch size, data size)
        :rtype: list of jnp.ndarray list
        """
        # ================================
        # - sample a batch from memory.
        #   It is possible to generate only the sampling indexes and call self.sample_by_index(...)
        # ================================
# [end-base-class-jax]

# =============================================================================

# [start-random-torch]
# import the memory class
from skrl.memories.torch import RandomMemory

# instantiate the memory (assumes there is a wrapped environment: env)
memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=env.device)
# [end-random-torch]


# [start-random-jax]
# import the memory class
from skrl.memories.jax import RandomMemory

# instantiate the memory (assumes there is a wrapped environment: env)
memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=env.device)
# [end-random-jax]
