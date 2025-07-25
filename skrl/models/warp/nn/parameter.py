import warp as wp

from .module import Module


class Parameter(Module):
    def __init__(self, data: wp.array, requires_grad: bool = True):
        super().__init__()
        self.data = data
        self.data.requires_grad = requires_grad

        self.register_parameter("data", self.data)
