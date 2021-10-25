from typing import Union, Tuple

import torch


class Noise():
    def __init__(self, device: str = "cuda:0") -> None:
        """
        Base class representing a noise

        Parameters
        ----------
        device: str, optional
            Device on which a torch tensor is or will be allocated (default: "cuda:0")
        """
        # TODO: what about parameters noise
        self.device = device

    def sample_like(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Sample a noise with the same size (shape) as the input tensor

        This method will call the sampling method as follows `.sample(tensor.size())`

        Parameters
        ----------
        tensor: torch.Tensor
            Input tensor used to determine output tensor size (shape)

        Returns
        -------
        torch.Tensor
            Sampled noise
        """
        return self.sample(tensor.size())

    def sample(self, size: Union[Tuple[int], torch.Size]) -> torch.Tensor:
        """
        Noise sampling method to be implemented by the inheriting classes

        Parameters
        ----------
        size: tuple or list of integers or torch.Size
            Shape of the sampled tensor

        Returns
        -------
        torch.Tensor
            Sampled noise
        """
        raise NotImplementedError("The sampling method (.sample()) is not implemented")