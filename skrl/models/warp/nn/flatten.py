import warp as wp

from .module import Module


class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: wp.array) -> wp.array:
        return input.reshape((input.shape[0], -1))
