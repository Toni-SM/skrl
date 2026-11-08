import pytest

import numpy as np
import warp as wp

import skrl.utils.framework.warp as warp_utils


@pytest.mark.parametrize(
    "condition_shape, input_shape, other_shape",
    [
        ((5,), (5,), (5,)),
        ((4, 3), (4, 3), (4, 3)),
        ((4, 1), (4, 3), (4, 3)),
        ((3,), (4, 3), (1, 3)),
        ((4, 1, 1), (1, 2, 3), (4, 2, 1)),
        ((2, 1, 3, 1), (2, 4, 3, 5), (5,)),
    ],
)
@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_where(capsys, condition_shape, input_shape, other_shape, dtype):
    condition = np.random.rand(*condition_shape) > 0.5
    x = (np.random.rand(*input_shape) * 100).astype(dtype)
    y = (np.random.rand(*other_shape) * 100).astype(dtype)

    output = warp_utils.where(wp.array(condition), wp.array(x), wp.array(y))
    expected = np.where(condition, x, y)
    assert output.shape == expected.shape
    assert output.dtype == wp.dtype_from_numpy(dtype)
    assert np.array_equal(output.numpy(), expected)


@pytest.mark.parametrize(
    "input, other, dtype",
    [
        (1.5, -2.5, wp.float32),
        (1, 0, wp.int32),
        (True, False, wp.bool),
        (1, 2.5, wp.float32),
        (np.float32(1.5), 0, wp.float32),
        (np.int64(3), 0, wp.int32),
        (np.bool_(True), False, wp.bool),
    ],
)
def test_where_scalars(capsys, input, other, dtype):
    condition = np.random.rand(4, 3) > 0.5

    output = warp_utils.where(wp.array(condition), input, other)
    assert output.dtype == dtype
    assert np.array_equal(output.numpy(), np.where(condition, input, other).astype(wp.dtype_to_numpy(dtype)))


def test_where_mixed_scalar_array(capsys):
    condition = np.random.rand(4, 3) > 0.5
    x = np.random.rand(4, 3).astype(np.float64)

    output = warp_utils.where(wp.array(condition), wp.array(x), 0)
    assert output.dtype == wp.float64
    assert np.array_equal(output.numpy(), np.where(condition, x, 0))

    output = warp_utils.where(wp.array(condition), -1.0, wp.array(x))
    assert output.dtype == wp.float64
    assert np.array_equal(output.numpy(), np.where(condition, -1.0, x))


def test_where_non_contiguous(capsys):
    condition = np.random.rand(4, 6) > 0.5
    x = np.random.rand(4, 6).astype(np.float32)
    y = np.random.rand(4, 6).astype(np.float32)

    output = warp_utils.where(wp.array(condition)[:, 1:4], wp.array(x)[:, ::2], wp.array(y)[:, 2:5])
    assert np.array_equal(output.numpy(), np.where(condition[:, 1:4], x[:, ::2], y[:, 2:5]))


def test_where_errors(capsys):
    x = wp.array(np.random.rand(4, 3).astype(np.float32))
    with pytest.raises(ValueError, match="dtype bool"):
        warp_utils.where(wp.ones((4, 3), dtype=wp.int32), x, x)
    with pytest.raises(ValueError, match="same dtype"):
        warp_utils.where(wp.ones((4, 3), dtype=wp.bool), x, wp.zeros((4, 3), dtype=wp.float64))
    with pytest.raises(ValueError):
        warp_utils.where(wp.ones((4, 2), dtype=wp.bool), x, x)
    with pytest.raises(ValueError, match="Floating-point scalar"):
        warp_utils.where(wp.ones((4, 3), dtype=wp.bool), 1.5, wp.zeros((4, 3), dtype=wp.int32))
    with pytest.raises(ValueError, match="Floating-point scalar"):
        warp_utils.where(wp.ones((4, 3), dtype=wp.bool), wp.zeros((4, 3), dtype=wp.bool), np.float32(0.5))
    if wp.get_cuda_device_count():
        cpu_condition = wp.ones((4, 3), dtype=wp.bool, device="cpu")
        with pytest.raises(ValueError, match="same as 'condition'"):
            warp_utils.where(cpu_condition, wp.zeros((4, 3), dtype=wp.float32, device="cuda:0"), 0.0)
        with pytest.raises(ValueError, match="same as 'condition'"):
            warp_utils.where(cpu_condition, 0.0, wp.zeros((4, 3), dtype=wp.float32, device="cuda:0"))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_where_device(capsys, device):
    if device.startswith("cuda") and not wp.get_cuda_device_count():
        pytest.skip("CUDA device not available")
    condition = np.random.rand(4, 3) > 0.5
    x = np.random.rand(4, 1).astype(np.float32)

    output = warp_utils.where(wp.array(condition, device=device), wp.array(x, device=device), 0.0)
    assert str(output.device) == device
    assert np.array_equal(output.numpy(), np.where(condition, x, 0.0))


def test_where_keeps_source_alive(capsys):
    condition = wp.array(np.random.rand(4, 3) > 0.5)
    x = np.random.rand(3).astype(np.float32)

    # the temporary array passed as `input` is only referenced by the broadcast view inside the function
    output = warp_utils.where(condition, wp.array(x), 0.0)
    assert np.array_equal(output.numpy(), np.where(condition.numpy(), x, 0.0))
