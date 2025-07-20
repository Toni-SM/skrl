from typing import Sequence

import math

import warp as wp

from ._generation import sequential_kernel_factory
from .config import tile_threads
from .module import Module


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        # register modules
        modules = args[0] if len(args) == 1 and isinstance(args[0], Sequence) else args
        for i, module in enumerate(modules):
            self.register_module(str(i), module)
        # get kernel and parameters
        _, factory, self._kernel_parameters = sequential_kernel_factory(self, tile_threads=tile_threads)
        self._input_size = self._kernel_parameters[0].shape[1]
        self._output_size = self._kernel_parameters[-1].shape[0]
        self._kernel = factory.create_kernel(self._input_size, self._output_size)
        # initialize output
        self.output = None

    def forward(self, input):
        shape = (input.shape[0], self._output_size)
        self.output = wp.empty(shape, dtype=wp.float32, device=self.device, requires_grad=True)
        wp.launch_tiled(
            self._kernel,
            dim=[math.ceil(input.shape[0] / tile_threads)],
            inputs=[
                input,
                self.output,
                *self._kernel_parameters,
            ],
            device=self.device,
            block_dim=tile_threads,
        )
        return self.output
