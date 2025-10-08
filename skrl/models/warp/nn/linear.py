import math

import numpy as np
import warp as wp

from .config import block_dim, nn_transposed_computation, tiles_dim_0, tiles_dim_1, tiles_dim_2
from .module import Module
from .parameter import Parameter


def create_kernel(in_features: int, out_features: int, transposed: bool):

    @wp.kernel
    def kernel(
        input: wp.array2d(dtype=float),
        weight: wp.array2d(dtype=float),
        bias: wp.array2d(dtype=float),
        output: wp.array2d(dtype=float),
    ):
        i, j = wp.tid()
        # compute the number of iteration steps for GEMM
        _in_features = weight.shape[
            1
        ]  # don't use `in_features` to have only one kernel definition for all Linear modules
        count = _in_features / tiles_dim_2
        if _in_features % tiles_dim_2 != 0:
            count += 1
        # static conditional (the generated code will contain only the branch that is taken)
        # - transposed input/output: (in_features, batch_size) -> (out_features, batch_size)
        if wp.static(transposed):
            sum = wp.tile_zeros(shape=(tiles_dim_0, tiles_dim_1), dtype=output.dtype)
            # - GEMM (weight * input)
            for k in range(0, count):
                tiled_weight = wp.tile_load(
                    weight, shape=(tiles_dim_0, tiles_dim_2), offset=(i * tiles_dim_0, k * tiles_dim_2)
                )
                x = wp.tile_load(input, shape=(tiles_dim_2, tiles_dim_1), offset=(k * tiles_dim_2, j * tiles_dim_1))
                wp.tile_matmul(tiled_weight, x, sum)
            # - bias
            tiled_bias = wp.tile_load(bias, shape=(tiles_dim_0, 1), offset=(i * tiles_dim_0, 0))
            sum += wp.tile_broadcast(tiled_bias, shape=(tiles_dim_0, tiles_dim_1))
            # store output
            wp.tile_store(output, sum, offset=(i * tiles_dim_0, j * tiles_dim_1))
        # - non-transposed input/output: (batch_size, in_features) -> (batch_size, out_features)
        else:
            sum = wp.tile_zeros(shape=(tiles_dim_1, tiles_dim_0), dtype=output.dtype)
            # - GEMM (weight * input)
            for k in range(0, count):
                tiled_weight = wp.tile_load(
                    weight, shape=(tiles_dim_1, tiles_dim_2), offset=(j * tiles_dim_1, k * tiles_dim_2)
                )
                x = wp.tile_transpose(
                    wp.tile_load(input, shape=(tiles_dim_0, tiles_dim_2), offset=(i * tiles_dim_0, k * tiles_dim_2))
                )
                wp.tile_matmul(tiled_weight, x, sum)
            # - bias
            tiled_bias = wp.tile_load(bias, shape=(tiles_dim_1, 1), offset=(j * tiles_dim_1, 0))
            sum += wp.tile_broadcast(tiled_bias, shape=(tiles_dim_1, tiles_dim_0))
            # store output
            wp.tile_store(output, wp.tile_transpose(sum), offset=(i * tiles_dim_0, j * tiles_dim_1))

    return kernel


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # create/register parameters
        # - weight
        self.weight = Parameter(
            wp.empty(shape=(self.out_features, self.in_features), dtype=wp.float32, device=self.device)
        )
        self.register_parameter("weight", self.weight)
        # - bias
        if bias:
            self.bias = Parameter(wp.empty(shape=(self.out_features, 1), dtype=wp.float32, device=self.device))
            self.register_parameter("bias", self.bias)
        else:
            self.bias = None
        self._init_module()

    def _init_module(self) -> None:
        # set default/initial values
        self.reset_parameters()
        # execution variables
        self._cache = {}
        self._kernel = create_kernel(self.in_features, self.out_features, transposed=nn_transposed_computation)

    def reset_parameters(self) -> None:
        # init parameters: sampling uniform(-1/sqrt(in_features), 1/sqrt(in_features)). \cite{he2015delving}
        bound = 1 / np.sqrt(self.in_features)
        # - weight
        value = np.random.uniform(-bound, bound, size=self.weight.data.shape)
        wp.copy(self.weight.data, wp.from_numpy(value, dtype=wp.float32))
        # - bias
        if self.bias:
            value = np.random.uniform(-bound, bound, size=self.bias.data.shape)
            wp.copy(self.bias.data, wp.from_numpy(value, dtype=wp.float32))

    def forward(self, input: wp.array) -> wp.array:
        shape = (
            (self.out_features, input.shape[1]) if nn_transposed_computation else (input.shape[0], self.out_features)
        )
        # cache output
        if shape not in self._cache:
            self._cache[shape] = wp.empty(shape, dtype=wp.float32, device=self.device, requires_grad=True)
        output = self._cache[shape]
        # launch kernel
        wp.launch_tiled(
            self._kernel,
            dim=[math.ceil(shape[0] / tiles_dim_0), math.ceil(shape[1] / tiles_dim_1)],
            inputs=[
                input,
                self.weight.data,
                self.bias.data,
            ],
            outputs=[output],
            device=self.device,
            block_dim=block_dim,
        )
        return output


class LazyLinear(Linear):
    def __init__(self, out_features: int, bias: bool = True):
        super().__init__(1, out_features, bias)
        self._initialized = False

    def forward(self, input: wp.array) -> wp.array:
        if not self._initialized:
            self.in_features = input.shape[0] if nn_transposed_computation else input.shape[1]
            self.weight.data = wp.empty(
                shape=(self.out_features, self.in_features), dtype=wp.float32, device=self.device, requires_grad=True
            )
            self._init_module()
            self._initialized = True
        return super().forward(input)
