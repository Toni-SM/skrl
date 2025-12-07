import math

import warp as wp

from .config import block_dim, tile_dim_0, tile_dim_1
from .module import Module


def create_kernel_elu(alpha: float):
    @wp.func
    def function(x: wp.float32):
        if x > 0.0:
            return x
        else:
            if wp.static(alpha == 1.0):
                return wp.exp(x) - 1.0
            else:
                return wp.static(alpha) * (wp.exp(x) - 1.0)

    @wp.kernel
    def kernel(
        input: wp.array2d(dtype=float),
        output: wp.array2d(dtype=float),
    ):
        i, j = wp.tid()
        # load input
        x = wp.tile_load(input, shape=(tile_dim_0, tile_dim_1), offset=(i * tile_dim_0, j * tile_dim_1))
        # computation
        x = wp.tile_map(function, x)
        # store output
        wp.tile_store(output, x, offset=(i * tile_dim_0, j * tile_dim_1))

    return kernel


def create_kernel_relu():
    @wp.func
    def function(x: wp.float32):
        return wp.max(x, 0.0)

    @wp.kernel
    def kernel(
        input: wp.array2d(dtype=float),
        output: wp.array2d(dtype=float),
    ):
        i, j = wp.tid()
        # load input
        x = wp.tile_load(input, shape=(tile_dim_0, tile_dim_1), offset=(i * tile_dim_0, j * tile_dim_1))
        # computation
        x = wp.tile_map(function, x)
        # store output
        wp.tile_store(output, x, offset=(i * tile_dim_0, j * tile_dim_1))

    return kernel


def create_kernel_tanh():
    @wp.func
    def function(x: wp.float32):
        return wp.tanh(x)

    @wp.kernel
    def kernel(
        input: wp.array2d(dtype=float),
        output: wp.array2d(dtype=float),
    ):
        i, j = wp.tid()
        # load input
        x = wp.tile_load(input, shape=(tile_dim_0, tile_dim_1), offset=(i * tile_dim_0, j * tile_dim_1))
        # computation
        x = wp.tile_map(function, x)
        # store output
        wp.tile_store(output, x, offset=(i * tile_dim_0, j * tile_dim_1))

    return kernel


class ELU(Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self._alpha = alpha
        # execution variables
        self._cache = {}
        self._kernel = create_kernel_elu(alpha)

    @property
    def alpha(self):
        return self._alpha

    def forward(self, input: wp.array) -> wp.array:
        shape = tuple(input.shape)
        # cache output
        if shape not in self._cache:
            self._cache[shape] = wp.empty(shape, dtype=wp.float32, device=self.device, requires_grad=True)
        output = self._cache[shape]
        # launch kernel
        wp.launch_tiled(
            self._kernel,
            dim=[math.ceil(shape[0] / tile_dim_0), math.ceil(shape[1] / tile_dim_1)],
            inputs=[input],
            outputs=[output],
            device=self.device,
            block_dim=block_dim,
        )
        return output


class ReLU(Module):
    def __init__(self):
        super().__init__()
        # execution variables
        self._cache = {}
        self._kernel = create_kernel_relu()

    def forward(self, input: wp.array) -> wp.array:
        shape = tuple(input.shape)
        # cache output
        if shape not in self._cache:
            self._cache[shape] = wp.empty(shape, dtype=wp.float32, device=self.device, requires_grad=True)
        output = self._cache[shape]
        # launch kernel
        wp.launch_tiled(
            self._kernel,
            dim=[math.ceil(shape[0] / tile_dim_0), math.ceil(shape[1] / tile_dim_1)],
            inputs=[input],
            outputs=[output],
            device=self.device,
            block_dim=block_dim,
        )
        return output


class Tanh(Module):
    def __init__(self):
        super().__init__()
        # execution variables
        self._cache = {}
        self._kernel = create_kernel_tanh()

    def forward(self, input: wp.array) -> wp.array:
        shape = tuple(input.shape)
        # cache output
        if shape not in self._cache:
            self._cache[shape] = wp.empty(shape, dtype=wp.float32, device=self.device, requires_grad=True)
        output = self._cache[shape]
        # launch kernel
        wp.launch_tiled(
            self._kernel,
            dim=[math.ceil(shape[0] / tile_dim_0), math.ceil(shape[1] / tile_dim_1)],
            inputs=[input],
            outputs=[output],
            device=self.device,
            block_dim=block_dim,
        )
        return output
