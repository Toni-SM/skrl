import warp as wp

from .module import Module, _Parameter


class Parameter(_Parameter, Module):
    def __init__(self, data: wp.array, requires_grad: bool = True):
        super().__init__()
        self.data = data
        self.data.requires_grad = requires_grad
