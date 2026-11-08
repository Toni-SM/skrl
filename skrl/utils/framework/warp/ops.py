from __future__ import annotations

from typing import Any

import numbers

import numpy as np
import warp as wp

from skrl.utils.framework.warp.types import Tensor


__all__ = ["clamp", "concatenate", "convert_to_numpy_in_place", "type_cast", "where"]


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


def _broadcast_to(array: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Get a (read-only) view of the array broadcast to the given shape (using zero strides)."""
    if array.shape == shape:
        return array
    offset = len(shape) - array.ndim
    strides = [0] * offset
    for size, stride, target_size in zip(array.shape, array.strides, shape[offset:]):
        strides.append(0 if size == 1 and target_size != 1 else stride)
    view = wp.array(ptr=array.ptr, shape=shape, strides=tuple(strides), dtype=array.dtype, device=array.device)
    view._ref = array  # keep the source array alive while the view exists
    return view


def where(condition: Tensor, input: Tensor | bool | int | float, other: Tensor | bool | int | float) -> Tensor:
    """Select elements from ``input`` or ``other`` depending on ``condition`` (analogous to ``torch.where``).

    The output is computed element-wise as ``input`` if ``condition`` is ``True`` and ``other`` otherwise.
    ``condition``, ``input`` and ``other`` are broadcast together following NumPy/PyTorch broadcasting rules.

    :param condition: Boolean array. When ``True`` (non-zero) yield ``input``, otherwise yield ``other``.
    :param input: Array or Python scalar selected at indices where ``condition`` is ``True``.
    :param other: Array or Python scalar selected at indices where ``condition`` is ``False``.

    :return: New array with the broadcast shape of the inputs.

    .. note::

        Unlike ``torch.where``, no type promotion is performed: ``input`` and ``other`` arrays must share the same
        data type, and scalars are cast to the data type of the array argument (if any).
        The output array is not differentiable.

    :raises ValueError: If ``condition`` is not a boolean array, if ``input`` and ``other`` arrays have different
        data types, if a floating-point scalar is mixed with a non floating-point array, if the arrays are not
        on the same device as ``condition``, or if the shapes can't be broadcast together.
    """
    if condition.dtype != wp.bool:
        raise ValueError(f"Expected 'condition' of dtype bool, got {condition.dtype}")
    device = condition.device
    for name, item in (("input", input), ("other", other)):
        if isinstance(item, wp.array) and item.device != device:
            raise ValueError(f"Expected '{name}' on device {device} (same as 'condition'), got {item.device}")
    # resolve the output data type
    if isinstance(input, wp.array) and isinstance(other, wp.array):
        if input.dtype != other.dtype:
            raise ValueError(f"Expected 'input' and 'other' with the same dtype, got {input.dtype} and {other.dtype}")
        dtype = input.dtype
    else:
        scalars = [item for item in (input, other) if not isinstance(item, wp.array)]
        has_float = any(not isinstance(item, (numbers.Integral, np.bool_)) for item in scalars)  # bool is Integral
        array = input if isinstance(input, wp.array) else other if isinstance(other, wp.array) else None
        if array is not None:
            dtype = array.dtype
            if has_float and not wp.types.type_is_float(dtype):
                raise ValueError(f"Floating-point scalar can't be safely cast to array dtype {dtype}")
        elif has_float:
            dtype = wp.float32
        elif all(isinstance(item, (bool, np.bool_)) for item in scalars):
            dtype = wp.bool
        else:
            dtype = wp.int32
    # convert scalars to single-element arrays
    input, other = [
        item if isinstance(item, wp.array) else wp.array([item], dtype=dtype, device=device) for item in (input, other)
    ]
    # broadcast the arrays to a common shape
    shape = np.broadcast_shapes(condition.shape, input.shape, other.shape)
    output = wp.empty(shape, dtype=dtype, device=device)
    wp.launch(
        _WHERE[len(shape)],
        dim=shape,
        inputs=[_broadcast_to(condition, shape), _broadcast_to(input, shape), _broadcast_to(other, shape)],
        outputs=[output],
        device=device,
    )
    return output


@wp.kernel(enable_backward=False)
def _where_1d(condition: wp.array1d[wp.bool], x: wp.array1d[Any], y: wp.array1d[Any], dst: wp.array1d[Any]):
    i = wp.tid()
    dst[i] = wp.where(condition[i], x[i], y[i])


@wp.kernel(enable_backward=False)
def _where_2d(condition: wp.array2d[wp.bool], x: wp.array2d[Any], y: wp.array2d[Any], dst: wp.array2d[Any]):
    i, j = wp.tid()
    dst[i, j] = wp.where(condition[i, j], x[i, j], y[i, j])


@wp.kernel(enable_backward=False)
def _where_3d(condition: wp.array3d[wp.bool], x: wp.array3d[Any], y: wp.array3d[Any], dst: wp.array3d[Any]):
    i, j, k = wp.tid()
    dst[i, j, k] = wp.where(condition[i, j, k], x[i, j, k], y[i, j, k])


@wp.kernel(enable_backward=False)
def _where_4d(condition: wp.array4d[wp.bool], x: wp.array4d[Any], y: wp.array4d[Any], dst: wp.array4d[Any]):
    i, j, k, l = wp.tid()
    dst[i, j, k, l] = wp.where(condition[i, j, k, l], x[i, j, k, l], y[i, j, k, l])


_WHERE = [None, _where_1d, _where_2d, _where_3d, _where_4d]
