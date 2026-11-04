from __future__ import annotations

from typing import Any

import warp as wp

from skrl.utils.framework.warp.types import Tensor


__all__ = ["clamp", "concatenate", "convert_to_numpy_in_place", "type_cast"]


@wp.kernel
def _clamp_1d(src: wp.array1d[Any], min: wp.array1d[Any], max: wp.array1d[Any], dst: wp.array1d[Any]):
    i = wp.tid()
    dst[i] = wp.clamp(src[i], src.dtype(min[i]), src.dtype(max[i]))  # pyright: ignore[reportCallIssue]


@wp.kernel
def _clamp_2d(src: wp.array2d[Any], min: wp.array1d[Any], max: wp.array1d[Any], dst: wp.array2d[Any]):
    i, j = wp.tid()
    dst[i, j] = wp.clamp(src[i, j], src.dtype(min[j]), src.dtype(max[j]))  # pyright: ignore[reportCallIssue]


@wp.kernel
def _clamp_3d(src: wp.array3d[Any], min: wp.array1d[Any], max: wp.array1d[Any], dst: wp.array3d[Any]):
    i, j, k = wp.tid()
    dst[i, j, k] = wp.clamp(src[i, j, k], src.dtype(min[k]), src.dtype(max[k]))  # pyright: ignore[reportCallIssue]


@wp.kernel
def _clamp_4d(src: wp.array4d[Any], min: wp.array1d[Any], max: wp.array1d[Any], dst: wp.array4d[Any]):
    i, j, k, l = wp.tid()
    dst[i, j, k, l] = wp.clamp(
        src[i, j, k, l], src.dtype(min[l]), src.dtype(max[l])  # pyright: ignore[reportCallIssue]
    )


_clamp = [None, _clamp_1d, _clamp_2d, _clamp_3d, _clamp_4d]


def clamp(array: Tensor, *, min: Tensor, max: Tensor, inplace: bool = False) -> Tensor:
    output = array if inplace else wp.empty_like(array)
    wp.launch(_clamp[array.ndim], dim=array.shape, inputs=[array, min, max], outputs=[output], device=array.device)
    return output


def concatenate(arrays: list[Tensor], *, axis: int = -1, dtype: type | None = None) -> Tensor:
    reference = arrays[0]
    dtype = reference.dtype if dtype is None else dtype
    shape = (reference.shape[0], sum([array.shape[1] for array in arrays]))
    output = wp.empty(shape, dtype=dtype, device=reference.device, requires_grad=reference.requires_grad)
    index = 0
    for array in arrays:
        next_index = index + array.shape[1]
        wp.copy(output[:, index:next_index], type_cast(array, dtype))
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


def type_cast(array: Tensor, dtype: type) -> Tensor:
    if array.dtype == dtype:
        return array
    output = wp.empty(array.shape, dtype=dtype, device=array.device, requires_grad=array.requires_grad)
    wp.launch(_TYPE_CAST[array.ndim], dim=array.shape, inputs=[array], outputs=[output], device=array.device)
    return output


@wp.kernel(enable_backward=False)
def _type_cast_1d(src: wp.array1d[Any], dst: wp.array1d[Any]):
    i = wp.tid()
    dst[i] = dst.dtype(src[i])


@wp.kernel(enable_backward=False)
def _type_cast_2d(src: wp.array2d[Any], dst: wp.array2d[Any]):
    i, j = wp.tid()
    dst[i, j] = dst.dtype(src[i, j])


@wp.kernel(enable_backward=False)
def _type_cast_3d(src: wp.array3d[Any], dst: wp.array3d[Any]):
    i, j, k = wp.tid()
    dst[i, j, k] = dst.dtype(src[i, j, k])


@wp.kernel(enable_backward=False)
def _type_cast_4d(src: wp.array4d[Any], dst: wp.array4d[Any]):
    i, j, k, l = wp.tid()
    dst[i, j, k, l] = dst.dtype(src[i, j, k, l])


_TYPE_CAST = [None, _type_cast_1d, _type_cast_2d, _type_cast_3d, _type_cast_4d]
