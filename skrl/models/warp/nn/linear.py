from typing import Sequence, Tuple

import numpy as np
import warp as wp

from .module import Module


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # create/register parameters
        # - weight
        self.weight = wp.empty(
            shape=(out_features, in_features), dtype=wp.float32, device=self.device, requires_grad=True
        )
        self.register_parameter("weight", self.weight)
        # - bias
        if bias:
            self.bias = wp.empty(shape=(out_features, 1), dtype=wp.float32, device=self.device, requires_grad=True)
        else:
            self.bias = None
        self.register_parameter("bias", self.bias)
        # set default/initial values
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # init parameters: sampling uniform(-1/sqrt(in_features), 1/sqrt(in_features)). \cite{he2015delving}
        bound = 1 / np.sqrt(self.weight.shape[1])
        # - weight
        value = np.random.uniform(-bound, bound, size=self.weight.shape)
        wp.copy(self.weight, wp.from_numpy(value, dtype=wp.float32))
        # - bias
        if self.bias:
            value = np.random.uniform(-bound, bound, size=self.bias.shape)
            wp.copy(self.bias, wp.from_numpy(value, dtype=wp.float32))

    def parse(self, uid: str) -> Tuple[str, Sequence[str], Sequence[str], Sequence[str], Sequence[str]]:
        # templates
        template_kernel = """
# Linear({in_features}, {out_features})
tiled_weight_{uid} = wp.tile_load(weight_{uid}, shape=({out_features}, {in_features}))
tiled_bias_{uid} = wp.tile_load(bias_{uid}, shape=({out_features}, 1))
{output} = wp.tile_matmul(tiled_weight_{uid}, {input}) + wp.tile_broadcast(tiled_bias_{uid}, shape=({out_features}, TILE_THREADS))
"""
        # generation
        functions = []
        kernel_parameters = [self.weight, self.bias]  # TODO: check when bias is None
        kernel_arguments = [
            "weight_{uid}: wp.array2d(dtype=float)".format(uid=uid),
            "bias_{uid}: wp.array2d(dtype=float)".format(uid=uid),
        ]
        kernel_definitions = [
            template_kernel.strip().format(
                uid=uid,
                input="{input}",
                output="{output}",
                in_features=self.in_features,
                out_features=self.out_features,
            ),
        ]
        return None, functions, kernel_parameters, kernel_arguments, kernel_definitions
