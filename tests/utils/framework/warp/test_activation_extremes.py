import pytest

import numpy as np
import warp as wp

import skrl.utils.framework.warp as warp_utils


VALUES = np.array([-1000, -100, -20, -1, 0, 1, 20, 100, 1000], dtype=np.float64)


def expected_values(name):
    exponential = np.exp(-np.abs(VALUES))
    sigmoid = np.where(VALUES >= 0, 1 / (1 + exponential), exponential / (1 + exponential))
    if name == "softplus":
        return np.logaddexp(0, VALUES), sigmoid
    return sigmoid, sigmoid * (1 - sigmoid)


@pytest.mark.parametrize("name", ["sigmoid", "softplus"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_extreme_activation_values(name, dtype):
    expected, _ = expected_values(name)
    tolerance = 1e-3 if dtype == np.float16 else 1e-6
    activate = getattr(warp_utils, name)
    with wp.ScopedDevice("cpu"):
        for shape in [(9,), (3, 3), (1, 3, 3), (1, 1, 3, 3)]:
            for inplace in [False, True]:
                array = wp.array(VALUES.astype(dtype).reshape(shape), device="cpu")
                output = activate(array, inplace=inplace)
                assert np.isfinite(output.numpy()).all()
                np.testing.assert_allclose(output.numpy().reshape(-1), expected, rtol=tolerance, atol=tolerance)
                assert (output is array) == inplace


@pytest.mark.parametrize("name", ["sigmoid", "softplus"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_extreme_activation_gradients(name, dtype):
    _, derivative = expected_values(name)
    tolerance = 1e-3 if dtype == np.float16 else 1e-6
    activate = getattr(warp_utils, name)
    with wp.ScopedDevice("cpu"):
        array = wp.array(VALUES.astype(dtype).reshape(3, 3), device="cpu", requires_grad=True)
        with wp.Tape() as tape:
            output = activate(array)
            loss = warp_utils.mean(output, dtype=wp.float64 if dtype == np.float64 else wp.float32)
        tape.backward(loss)
        gradients = array.grad.numpy().reshape(-1)
        assert np.isfinite(gradients).all()
        np.testing.assert_allclose(gradients, derivative / VALUES.size, rtol=tolerance, atol=tolerance)
