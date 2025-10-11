from __future__ import annotations

import warp as wp


__all__ = ["clamp", "concatenate", "convert_to_numpy_in_place", "type_cast"]


@wp.kernel
def _clamp_1d(src: wp.array(ndim=1), min: wp.array(ndim=1), max: wp.array(ndim=1), dst: wp.array(ndim=1)):
    i = wp.tid()
    dst[i] = wp.clamp(src[i], src.dtype(min[i]), src.dtype(max[i]))


@wp.kernel
def _clamp_2d(src: wp.array(ndim=2), min: wp.array(ndim=1), max: wp.array(ndim=1), dst: wp.array(ndim=2)):
    i, j = wp.tid()
    dst[i, j] = wp.clamp(src[i, j], src.dtype(min[j]), src.dtype(max[j]))


@wp.kernel
def _clamp_3d(src: wp.array(ndim=3), min: wp.array(ndim=1), max: wp.array(ndim=1), dst: wp.array(ndim=3)):
    i, j, k = wp.tid()
    dst[i, j, k] = wp.clamp(src[i, j, k], src.dtype(min[k]), src.dtype(max[k]))


@wp.kernel
def _clamp_4d(src: wp.array(ndim=4), min: wp.array(ndim=1), max: wp.array(ndim=1), dst: wp.array(ndim=4)):
    i, j, k, l = wp.tid()
    dst[i, j, k, l] = wp.clamp(src[i, j, k, l], src.dtype(min[l]), src.dtype(max[l]))


_clamp = [None, _clamp_1d, _clamp_2d, _clamp_3d, _clamp_4d]


def clamp(array: wp.array, *, min: wp.array, max: wp.array, inplace: bool = False) -> wp.array:
    output = array if inplace else wp.empty_like(array)
    wp.launch(_clamp[array.ndim], dim=array.shape, inputs=[array, min, max], outputs=[output], device=array.device)
    return output


def concatenate(arrays: list[wp.array], *, axis: int = -1) -> wp.array:
    reference = arrays[0]
    shape = (reference.shape[0], sum([array.shape[1] for array in arrays]))
    output = wp.empty(shape, dtype=reference.dtype, device=reference.device, requires_grad=reference.requires_grad)
    index = 0
    for array in arrays:
        next_index = index + array.shape[1]
        wp.copy(output[:, index:next_index], array)
        index = next_index
    return output


def convert_to_numpy_in_place(src):
    if isinstance(src, dict):
        for k, v in src.items():
            if isinstance(v, wp.array):
                src[k] = v.numpy()
            elif isinstance(v, dict):
                convert_to_numpy_in_place(v)
    elif isinstance(src, wp.array):
        return src.numpy()
    return src


def type_cast(array: wp.array, dtype: type) -> wp.array:
    if array.dtype == dtype:
        return array
    output = wp.empty(array.shape, dtype=dtype, device=array.device, requires_grad=array.requires_grad)
    wp.launch(_TYPE_CAST[array.ndim], dim=array.shape, inputs=[array], outputs=[output], device=array.device)
    return output


@wp.kernel(enable_backward=False)
def _type_cast_1d(src: wp.array(ndim=1), dst: wp.array(ndim=1)):
    i = wp.tid()
    dst[i] = dst.dtype(src[i])


@wp.kernel(enable_backward=False)
def _type_cast_2d(src: wp.array(ndim=2), dst: wp.array(ndim=2)):
    i, j = wp.tid()
    dst[i, j] = dst.dtype(src[i, j])


@wp.kernel(enable_backward=False)
def _type_cast_3d(src: wp.array(ndim=3), dst: wp.array(ndim=3)):
    i, j, k = wp.tid()
    dst[i, j, k] = dst.dtype(src[i, j, k])


@wp.kernel(enable_backward=False)
def _type_cast_4d(src: wp.array(ndim=4), dst: wp.array(ndim=4)):
    i, j, k, l = wp.tid()
    dst[i, j, k, l] = dst.dtype(src[i, j, k, l])


_TYPE_CAST = [None, _type_cast_1d, _type_cast_2d, _type_cast_3d, _type_cast_4d]
