from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp


__all__ = [
    "scalar_mul",
    "elu",
    "leaky_relu",
    "relu",
    "selu",
    "sigmoid",
    "softplus",
    "softsign",
    "tanh",
    "mean",
    "var",
    "std",
]


def scalar_mul(array: wp.array, scalar: int | float, inplace: bool = False) -> wp.array:
    output = (
        array
        if inplace
        else wp.empty(array.shape, dtype=array.dtype, device=array.device, requires_grad=array.requires_grad)
    )
    wp.launch(_scalar_mul, dim=array.shape, inputs=[output, array, scalar], device=array.device)
    return output


def mean(array: wp.array, *, dtype: type = wp.float32) -> wp.array:
    output = wp.zeros((1,), dtype=dtype, device=array.device, requires_grad=array.requires_grad)
    wp.launch(
        _MEAN[array.ndim],
        dim=array.shape,
        inputs=[array, np.prod(array.shape).item()],
        outputs=[output],
        device=array.device,
    )
    return output


def var(array: wp.array, *, dtype: type = wp.float32, correction: int = 1) -> wp.array:
    output = wp.zeros((1,), dtype=dtype, device=array.device, requires_grad=array.requires_grad)
    wp.launch(
        _VAR[array.ndim],
        dim=array.shape,
        inputs=[array, mean(array, dtype=dtype), np.prod(array.shape).item() - correction],
        outputs=[output],
        device=array.device,
    )
    return output


def std(array: wp.array, *, dtype: type = wp.float32, correction: int = 1) -> wp.array:
    _var = var(array, dtype=dtype, correction=correction)
    output = wp.zeros((1,), dtype=dtype, device=array.device, requires_grad=True) if array.requires_grad else _var
    wp.launch(_std, dim=1, inputs=[_var], outputs=[output], device=array.device)
    return output


def elu(array: wp.array, *, alpha: float = 1.0, inplace: bool = False) -> wp.array:
    output = array if inplace else wp.empty(array.shape, dtype=array.dtype, requires_grad=array.requires_grad)
    wp.launch(_ELU[array.ndim], dim=array.shape, inputs=[array, alpha], outputs=[output], device=array.device)
    return output


def leaky_relu(array: wp.array, *, negative_slope: float = 0.01, inplace: bool = False) -> wp.array:
    output = array if inplace else wp.empty(array.shape, dtype=array.dtype, requires_grad=array.requires_grad)
    wp.launch(
        _LEAKY_RELU[array.ndim], dim=array.shape, inputs=[array, negative_slope], outputs=[output], device=array.device
    )
    return output


def relu(array: wp.array, *, inplace: bool = False) -> wp.array:
    output = array if inplace else wp.empty(array.shape, dtype=array.dtype, requires_grad=array.requires_grad)
    wp.launch(_RELU[array.ndim], dim=array.shape, inputs=[array], outputs=[output], device=array.device)
    return output


def selu(array: wp.array, *, inplace: bool = False) -> wp.array:
    output = array if inplace else wp.empty(array.shape, dtype=array.dtype, requires_grad=array.requires_grad)
    wp.launch(_SELU[array.ndim], dim=array.shape, inputs=[array], outputs=[output], device=array.device)
    return output


def sigmoid(array: wp.array, *, inplace: bool = False) -> wp.array:
    output = array if inplace else wp.empty(array.shape, dtype=array.dtype, requires_grad=array.requires_grad)
    wp.launch(_SIGMOID[array.ndim], dim=array.shape, inputs=[array], outputs=[output], device=array.device)
    return output


def softplus(array: wp.array, *, inplace: bool = False) -> wp.array:
    output = array if inplace else wp.empty(array.shape, dtype=array.dtype, requires_grad=array.requires_grad)
    wp.launch(_SOFTPLUS[array.ndim], dim=array.shape, inputs=[array], outputs=[output], device=array.device)
    return output


def softsign(array: wp.array, *, inplace: bool = False) -> wp.array:
    output = array if inplace else wp.empty(array.shape, dtype=array.dtype, requires_grad=array.requires_grad)
    wp.launch(_SOFTSIGN[array.ndim], dim=array.shape, inputs=[array], outputs=[output], device=array.device)
    return output


def tanh(array: wp.array, *, inplace: bool = False) -> wp.array:
    output = array if inplace else wp.empty(array.shape, dtype=array.dtype, requires_grad=array.requires_grad)
    wp.launch(_TANH[array.ndim], dim=array.shape, inputs=[array], outputs=[output], device=array.device)
    return output


# Warp functions


@wp.func
def _f_elu(x: Any, alpha: Any):
    if x >= type(x)(0.0):
        return x
    else:
        return type(x)(alpha) * (wp.exp(x) - type(x)(1.0))


@wp.func
def _f_leaky_relu(x: Any, negative_slope: Any):
    if x >= type(x)(0.0):
        return x
    else:
        return type(x)(negative_slope) * x


@wp.func
def _f_relu(x: Any):
    return wp.max(x, type(x)(0.0))


@wp.func
def _f_selu(x: Any):
    alpha = type(x)(1.6732632423543772848170429916717)
    scale = type(x)(1.0507009873554804934193349852946)
    return scale * _f_elu(x, alpha)


@wp.func
def _f_sigmoid(x: Any):
    return type(x)(1.0) / (type(x)(1.0) + wp.exp(-x))


@wp.func
def _f_softplus(x: Any):
    return wp.log(type(x)(1.0) + wp.exp(x))


@wp.func
def _f_softsign(x: Any):
    return x / (type(x)(1.0) + wp.abs(x))


@wp.func
def _f_tanh(x: Any):
    return wp.tanh(x)


# Warp kernels


@wp.kernel
def _mean_1d(src: wp.array(ndim=1), n: int, dst: wp.array(ndim=1)):
    i = wp.tid()
    wp.atomic_add(dst, 0, dst.dtype(src[i]) / dst.dtype(n))


@wp.kernel
def _mean_2d(src: wp.array(ndim=2), n: int, dst: wp.array(ndim=1)):
    i, j = wp.tid()
    wp.atomic_add(dst, 0, dst.dtype(src[i, j]) / dst.dtype(n))


@wp.kernel
def _mean_3d(src: wp.array(ndim=3), n: int, dst: wp.array(ndim=1)):
    i, j, k = wp.tid()
    wp.atomic_add(dst, 0, dst.dtype(src[i, j, k]) / dst.dtype(n))


@wp.kernel
def _mean_4d(src: wp.array(ndim=4), n: int, dst: wp.array(ndim=1)):
    i, j, k, l = wp.tid()
    wp.atomic_add(dst, 0, dst.dtype(src[i, j, k, l]) / dst.dtype(n))


_MEAN = [None, _mean_1d, _mean_2d, _mean_3d, _mean_4d]


@wp.kernel
def _var_1d(src: wp.array(ndim=1), mean: wp.array(ndim=1), n: int, dst: wp.array(ndim=1)):
    i = wp.tid()
    wp.atomic_add(dst, 0, wp.pow(dst.dtype(src[i]) - mean[0], 2.0) / dst.dtype(n))


@wp.kernel
def _var_2d(src: wp.array(ndim=2), mean: wp.array(ndim=1), n: int, dst: wp.array(ndim=1)):
    i, j = wp.tid()
    wp.atomic_add(dst, 0, wp.pow(dst.dtype(src[i, j]) - mean[0], 2.0) / dst.dtype(n))


@wp.kernel
def _var_3d(src: wp.array(ndim=3), mean: wp.array(ndim=1), n: int, dst: wp.array(ndim=1)):
    i, j, k = wp.tid()
    wp.atomic_add(dst, 0, wp.pow(dst.dtype(src[i, j, k]) - mean[0], 2.0) / dst.dtype(n))


@wp.kernel
def _var_4d(src: wp.array(ndim=4), mean: wp.array(ndim=1), n: int, dst: wp.array(ndim=1)):
    i, j, k, l = wp.tid()
    wp.atomic_add(dst, 0, wp.pow(dst.dtype(src[i, j, k, l]) - mean[0], 2.0) / dst.dtype(n))


_VAR = [None, _var_1d, _var_2d, _var_3d, _var_4d]


@wp.kernel
def _elu_1d(src: wp.array(ndim=1), alpha: Any, dst: wp.array(ndim=1)):
    i = wp.tid()
    dst[i] = _f_elu(src[i], alpha)


@wp.kernel
def _elu_2d(src: wp.array(ndim=2), alpha: Any, dst: wp.array(ndim=2)):
    i, j = wp.tid()
    dst[i, j] = _f_elu(src[i, j], alpha)


@wp.kernel
def _elu_3d(src: wp.array(ndim=3), alpha: Any, dst: wp.array(ndim=3)):
    i, j, k = wp.tid()
    dst[i, j, k] = _f_elu(src[i, j, k], alpha)


@wp.kernel
def _elu_4d(src: wp.array(ndim=4), alpha: Any, dst: wp.array(ndim=4)):
    i, j, k, l = wp.tid()
    dst[i, j, k, l] = _f_elu(src[i, j, k, l], alpha)


@wp.kernel
def _leaky_relu_1d(src: wp.array(ndim=1), negative_slope: Any, dst: wp.array(ndim=1)):
    i = wp.tid()
    dst[i] = _f_leaky_relu(src[i], negative_slope)


@wp.kernel
def _leaky_relu_2d(src: wp.array(ndim=2), negative_slope: Any, dst: wp.array(ndim=2)):
    i, j = wp.tid()
    dst[i, j] = _f_leaky_relu(src[i, j], negative_slope)


@wp.kernel
def _leaky_relu_3d(src: wp.array(ndim=3), negative_slope: Any, dst: wp.array(ndim=3)):
    i, j, k = wp.tid()
    dst[i, j, k] = _f_leaky_relu(src[i, j, k], negative_slope)


@wp.kernel
def _leaky_relu_4d(src: wp.array(ndim=4), negative_slope: Any, dst: wp.array(ndim=4)):
    i, j, k, l = wp.tid()
    dst[i, j, k, l] = _f_leaky_relu(src[i, j, k, l], negative_slope)


@wp.kernel
def _relu_1d(src: wp.array(ndim=1), dst: wp.array(ndim=1)):
    i = wp.tid()
    dst[i] = _f_relu(src[i])


@wp.kernel
def _relu_2d(src: wp.array(ndim=2), dst: wp.array(ndim=2)):
    i, j = wp.tid()
    dst[i, j] = _f_relu(src[i, j])


@wp.kernel
def _relu_3d(src: wp.array(ndim=3), dst: wp.array(ndim=3)):
    i, j, k = wp.tid()
    dst[i, j, k] = _f_relu(src[i, j, k])


@wp.kernel
def _relu_4d(src: wp.array(ndim=4), dst: wp.array(ndim=4)):
    i, j, k, l = wp.tid()
    dst[i, j, k, l] = _f_relu(src[i, j, k, l])


@wp.kernel
def _selu_1d(src: wp.array(ndim=1), dst: wp.array(ndim=1)):
    i = wp.tid()
    dst[i] = _f_selu(src[i])


@wp.kernel
def _selu_2d(src: wp.array(ndim=2), dst: wp.array(ndim=2)):
    i, j = wp.tid()
    dst[i, j] = _f_selu(src[i, j])


@wp.kernel
def _selu_3d(src: wp.array(ndim=3), dst: wp.array(ndim=3)):
    i, j, k = wp.tid()
    dst[i, j, k] = _f_selu(src[i, j, k])


@wp.kernel
def _selu_4d(src: wp.array(ndim=4), dst: wp.array(ndim=4)):
    i, j, k, l = wp.tid()
    dst[i, j, k, l] = _f_selu(src[i, j, k, l])


@wp.kernel
def _sigmoid_1d(src: wp.array(ndim=1), dst: wp.array(ndim=1)):
    i = wp.tid()
    dst[i] = _f_sigmoid(src[i])


@wp.kernel
def _sigmoid_2d(src: wp.array(ndim=2), dst: wp.array(ndim=2)):
    i, j = wp.tid()
    dst[i, j] = _f_sigmoid(src[i, j])


@wp.kernel
def _sigmoid_3d(src: wp.array(ndim=3), dst: wp.array(ndim=3)):
    i, j, k = wp.tid()
    dst[i, j, k] = _f_sigmoid(src[i, j, k])


@wp.kernel
def _sigmoid_4d(src: wp.array(ndim=4), dst: wp.array(ndim=4)):
    i, j, k, l = wp.tid()
    dst[i, j, k, l] = _f_sigmoid(src[i, j, k, l])


@wp.kernel
def _softplus_1d(src: wp.array(ndim=1), dst: wp.array(ndim=1)):
    i = wp.tid()
    dst[i] = _f_softplus(src[i])


@wp.kernel
def _softplus_2d(src: wp.array(ndim=2), dst: wp.array(ndim=2)):
    i, j = wp.tid()
    dst[i, j] = _f_softplus(src[i, j])


@wp.kernel
def _softplus_3d(src: wp.array(ndim=3), dst: wp.array(ndim=3)):
    i, j, k = wp.tid()
    dst[i, j, k] = _f_softplus(src[i, j, k])


@wp.kernel
def _softplus_4d(src: wp.array(ndim=4), dst: wp.array(ndim=4)):
    i, j, k, l = wp.tid()
    dst[i, j, k, l] = _f_softplus(src[i, j, k, l])


@wp.kernel
def _softsign_1d(src: wp.array(ndim=1), dst: wp.array(ndim=1)):
    i = wp.tid()
    dst[i] = _f_softsign(src[i])


@wp.kernel
def _softsign_2d(src: wp.array(ndim=2), dst: wp.array(ndim=2)):
    i, j = wp.tid()
    dst[i, j] = _f_softsign(src[i, j])


@wp.kernel
def _softsign_3d(src: wp.array(ndim=3), dst: wp.array(ndim=3)):
    i, j, k = wp.tid()
    dst[i, j, k] = _f_softsign(src[i, j, k])


@wp.kernel
def _softsign_4d(src: wp.array(ndim=4), dst: wp.array(ndim=4)):
    i, j, k, l = wp.tid()
    dst[i, j, k, l] = _f_softsign(src[i, j, k, l])


@wp.kernel
def _tanh_1d(src: wp.array(ndim=1), dst: wp.array(ndim=1)):
    i = wp.tid()
    dst[i] = _f_tanh(src[i])


@wp.kernel
def _tanh_2d(src: wp.array(ndim=2), dst: wp.array(ndim=2)):
    i, j = wp.tid()
    dst[i, j] = _f_tanh(src[i, j])


@wp.kernel
def _tanh_3d(src: wp.array(ndim=3), dst: wp.array(ndim=3)):
    i, j, k = wp.tid()
    dst[i, j, k] = _f_tanh(src[i, j, k])


@wp.kernel
def _tanh_4d(src: wp.array(ndim=4), dst: wp.array(ndim=4)):
    i, j, k, l = wp.tid()
    dst[i, j, k, l] = _f_tanh(src[i, j, k, l])


@wp.kernel
def _std(src: wp.array(ndim=1), dst: wp.array(ndim=1)):
    dst[0] = wp.sqrt(src[0])


@wp.kernel
def _scalar_mul(dst: wp.array2d(dtype=Any), src: wp.array2d(dtype=Any), scalar: Any):
    i, j = wp.tid()
    dst[i, j] = src[i, j] * dst.dtype(scalar)


_ELU = [None, _elu_1d, _elu_2d, _elu_3d, _elu_4d]
_LEAKY_RELU = [None, _leaky_relu_1d, _leaky_relu_2d, _leaky_relu_3d, _leaky_relu_4d]
_RELU = [None, _relu_1d, _relu_2d, _relu_3d, _relu_4d]
_SELU = [None, _selu_1d, _selu_2d, _selu_3d, _selu_4d]
_SOFTSIGN = [None, _softsign_1d, _softsign_2d, _softsign_3d, _softsign_4d]
_SOFTPLUS = [None, _softplus_1d, _softplus_2d, _softplus_3d, _softplus_4d]
_SIGMOID = [None, _sigmoid_1d, _sigmoid_2d, _sigmoid_3d, _sigmoid_4d]
_TANH = [None, _tanh_1d, _tanh_2d, _tanh_3d, _tanh_4d]
