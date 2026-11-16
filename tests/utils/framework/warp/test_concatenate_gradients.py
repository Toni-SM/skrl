import pytest

import numpy as np
import warp as wp

from skrl.utils.framework.warp import concatenate, mean


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("requires_grad", [(False, True), (True, False), (True, True), (False, False)])
def test_concatenation_preserves_gradients_from_any_input(batch_size, requires_grad):
    first_values = np.arange(2 * batch_size, dtype=np.float32).reshape(batch_size, 2)
    second_values = np.arange(3 * batch_size, dtype=np.float32).reshape(batch_size, 3) + 0.5
    first = wp.array(first_values, device="cpu", requires_grad=requires_grad[0])
    second = wp.array(second_values, device="cpu", requires_grad=requires_grad[1])
    with wp.Tape() as tape:
        output = concatenate([first, second])
        loss = mean(output)
    np.testing.assert_array_equal(output.numpy(), np.concatenate([first_values, second_values], axis=-1))
    assert output.requires_grad == any(requires_grad)
    if any(requires_grad):
        tape.backward(loss)
        for array, enabled in zip([first, second], requires_grad):
            if enabled:
                np.testing.assert_allclose(array.grad.numpy(), np.full(array.shape, 1 / (5 * batch_size)), rtol=1e-6)
            else:
                assert array.grad is None
