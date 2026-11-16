import pytest

import numpy as np
import warp as wp

from skrl.utils.framework.warp import mean, type_cast


@pytest.mark.parametrize(
    "source_dtype,target_dtype",
    [
        (np.float16, wp.float32),
        (np.float16, wp.float64),
        (np.float32, wp.float16),
        (np.float32, wp.float64),
        (np.float64, wp.float16),
        (np.float64, wp.float32),
    ],
)
def test_float_type_cast_preserves_gradients(source_dtype, target_dtype):
    values = np.array([0.5, 1.25, -2.0, 3.0], dtype=source_dtype)
    for shape in [(4,), (2, 2), (1, 2, 2), (1, 1, 2, 2)]:
        source = wp.array(values.reshape(shape), device="cpu", requires_grad=True)
        with wp.Tape() as tape:
            converted = type_cast(source, target_dtype)
            loss = mean(converted)
        tape.backward(loss)
        assert converted.requires_grad
        np.testing.assert_array_equal(converted.numpy().reshape(-1), values)
        np.testing.assert_allclose(source.grad.numpy(), np.full(shape, 0.25), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", [wp.float16, wp.float32, wp.float64])
def test_identity_cast_preserves_the_original_array(dtype):
    source = wp.array([1.0, 2.0], dtype=dtype, device="cpu", requires_grad=True)
    assert type_cast(source, dtype) is source


def test_integer_casts_preserve_values_without_requiring_gradients():
    values = np.array([0, 1, -2, 3], dtype=np.int32)
    for shape in [(4,), (2, 2), (1, 2, 2), (1, 1, 2, 2)]:
        source = wp.array(values.reshape(shape), device="cpu")
        converted = type_cast(source, wp.float32)
        assert not converted.requires_grad
        np.testing.assert_array_equal(converted.numpy().reshape(-1), values)
        restored = type_cast(converted, wp.int32)
        np.testing.assert_array_equal(restored.numpy().reshape(-1), values)
