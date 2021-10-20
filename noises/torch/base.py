from typing import Union

import torch


class Noise():
    def __init__(self, device: str = "cuda:0") -> None:
        """
        Base class that represent a noise

        Parameters
        ----------
        device: str, optional
            Device on which a torch tensor is or will be allocated (default: "cuda:0")
        """
        self.device = device

    def sample(self, shape: Union[tuple[int], list[int], torch.Size]) -> torch.Tensor:
        """
        Sample noise

        Parameters
        ----------
        shape: torch.Size or tuple or list of ints
            Number of element on each dimension

        Returns
        -------
        tuple
            Sampled noise
        """
        raise NotImplementedError("The sampling method (.sample()) is not implemented")