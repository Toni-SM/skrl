from typing import List

import warp as wp


@wp.kernel
def _copy_j(dst: wp.array2d(dtype=float), src: wp.array2d(), index: int):
    i, j = wp.tid()
    dst[i, index + j] = wp.float(src[i, j])


@wp.kernel
def _scalar_multiplication(dst: wp.array2d(dtype=float), src: wp.array2d(), k: float):
    i, j = wp.tid()
    dst[i, j] = k * wp.float(src[i, j])


def concatenate(arrays: List[wp.array], axis=None) -> wp.array:
    ref = arrays[0]
    shape = (ref.shape[0], sum([array.shape[1] for array in arrays]))
    output = wp.empty(shape, dtype=wp.float32, device=ref.device, requires_grad=ref.requires_grad)
    index = 0
    for array in arrays:
        # warp proposal: wp.copy should allow to copy by axis (not only by rows)
        wp.launch(_copy_j, dim=array.shape, inputs=[output, array, index], device=ref.device)
        index += array.shape[1]
    return output


def scalar_multiplication(array: wp.array, k: float) -> wp.array:
    output = wp.empty(array.shape, dtype=wp.float32, device=array.device, requires_grad=array.requires_grad)
    wp.launch(_scalar_multiplication, dim=array.shape, inputs=[output, array, k], device=array.device)
    return output
