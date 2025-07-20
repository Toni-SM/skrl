from typing import Any, Union

import warp as wp


__all__ = ["scalar_mul"]


@wp.kernel
def _scalar_mul(dst: wp.array2d(dtype=Any), src: wp.array2d(dtype=Any), scalar: Any):
    i, j = wp.tid()
    dst[i, j] = src[i, j] * dst.dtype(scalar)


def scalar_mul(array: wp.array, scalar: Union[int, float], inplace: bool = False) -> wp.array:
    output = (
        array
        if inplace
        else wp.empty(array.shape, dtype=array.dtype, device=array.device, requires_grad=array.requires_grad)
    )
    wp.launch(_scalar_mul, dim=array.shape, inputs=[output, array, scalar], device=array.device)
    return output
