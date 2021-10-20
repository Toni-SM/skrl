import torch
import numpy as np

from .base import Memory


class RandomMemory(Memory):
    def __init__(self, memory_size: int, num_envs: int = 1, device: str = "cuda:0", preallocate: bool = True) -> None:
        """
        Random sampling memory

        Sample a batch from the memory randomly

        Parameters
        ----------
        memory_size: int
            Maximum number of elements in the first dimension of each internal tensor
        num_envs: int
            Number of parallel environments
        device: str, optional
            Device on which a torch tensor is or will be allocated (default: "cuda:0")
        preallocate: bool, optional
            If true, preallocate memory for efficient use (default: True)
        """
        super().__init__(memory_size=memory_size, num_envs=num_envs, device=device, preallocate=preallocate)

    def sample(self, batch_size: int, names: list[str] = []) -> list[torch.Tensor]:
        # generate random indexes
        indexes =  np.random.choice(len(self), size=batch_size, replace=True)
        
        return self.sample_by_index(indexes=indexes, names=names)
