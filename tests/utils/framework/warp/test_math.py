import hypothesis
import hypothesis.strategies as st
import pytest

import numpy as np
import warp as wp

import skrl.utils.framework.warp as warp_utils


@hypothesis.given(shape=st.lists(st.integers(min_value=1, max_value=10), min_size=4, max_size=4))
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_mean(capsys, ndim, dtype, shape):
    sample = (np.random.rand(*shape[:ndim]) * 100).astype(dtype)
    array = wp.array(sample)

    value = warp_utils.mean(array)
    assert np.allclose(value.numpy().item(), np.mean(sample), atol=1e-05, rtol=1e-03)


@hypothesis.given(shape=st.lists(st.integers(min_value=1, max_value=10), min_size=4, max_size=4))
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_var(capsys, ndim, dtype, shape):
    sample = (np.random.rand(*shape[:ndim]) * 100).astype(dtype)
    array = wp.array(sample)

    value = warp_utils.var(array, correction=0)
    assert np.allclose(value.numpy().item(), np.var(sample, ddof=0), atol=1e-05, rtol=1e-03)

    value = warp_utils.var(array, correction=1)
    assert np.allclose(value.numpy().item(), np.var(sample, ddof=1), atol=1e-05, rtol=1e-03, equal_nan=True)


@hypothesis.given(shape=st.lists(st.integers(min_value=1, max_value=10), min_size=4, max_size=4))
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_std(capsys, ndim, dtype, shape):
    sample = (np.random.rand(*shape[:ndim]) * 100).astype(dtype)
    array = wp.array(sample)

    value = warp_utils.std(array, correction=0)
    assert np.allclose(value.numpy().item(), np.std(sample, ddof=0), atol=1e-05, rtol=1e-03)

    value = warp_utils.std(array, correction=1)
    assert np.allclose(value.numpy().item(), np.std(sample, ddof=1), atol=1e-05, rtol=1e-03, equal_nan=True)


@hypothesis.given(
    shape=st.lists(st.integers(min_value=1, max_value=10), min_size=4, max_size=4),
    alpha=st.floats(min_value=0.0, max_value=10.0),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("inplace", [True, False])
def test_elu(capsys, ndim, dtype, inplace, shape, alpha):
    def elu(x):
        return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

    sample = (10 * (2 * np.random.rand(*shape[:ndim]) - 1)).astype(dtype)
    input = wp.array(sample)
    output = warp_utils.elu(input, alpha=alpha, inplace=inplace)
    assert output is input if inplace else output is not input
    assert np.allclose(output.numpy(), elu(sample), atol=1e-05, rtol=1e-03)


@hypothesis.given(
    shape=st.lists(st.integers(min_value=1, max_value=10), min_size=4, max_size=4),
    negative_slope=st.floats(min_value=0.0, max_value=10.0),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("inplace", [True, False])
def test_leaky_relu(capsys, ndim, dtype, inplace, shape, negative_slope):
    def leaky_relu(x):
        return np.where(x >= 0, x, negative_slope * x)

    sample = (10 * (2 * np.random.rand(*shape[:ndim]) - 1)).astype(dtype)
    input = wp.array(sample)
    output = warp_utils.leaky_relu(input, negative_slope=negative_slope, inplace=inplace)
    assert output is input if inplace else output is not input
    assert np.allclose(output.numpy(), leaky_relu(sample), atol=1e-05, rtol=1e-03)


@hypothesis.given(shape=st.lists(st.integers(min_value=1, max_value=10), min_size=4, max_size=4))
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("inplace", [True, False])
def test_relu(capsys, ndim, dtype, inplace, shape):
    def relu(x):
        return np.where(x >= 0, x, 0)

    sample = (10 * (2 * np.random.rand(*shape[:ndim]) - 1)).astype(dtype)
    input = wp.array(sample)
    output = warp_utils.relu(input, inplace=inplace)
    assert output is input if inplace else output is not input
    assert np.allclose(output.numpy(), relu(sample), atol=1e-05, rtol=1e-03)


@hypothesis.given(shape=st.lists(st.integers(min_value=1, max_value=10), min_size=4, max_size=4))
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("inplace", [True, False])
def test_selu(capsys, ndim, dtype, inplace, shape):
    def selu(x):
        return 1.0507009873554804934193349852946 * np.where(
            x >= 0, x, 1.6732632423543772848170429916717 * (np.exp(x) - 1)
        )

    sample = (10 * (2 * np.random.rand(*shape[:ndim]) - 1)).astype(dtype)
    input = wp.array(sample)
    output = warp_utils.selu(input, inplace=inplace)
    assert output is input if inplace else output is not input
    assert np.allclose(output.numpy(), selu(sample), atol=1e-05, rtol=1e-03)


@hypothesis.given(shape=st.lists(st.integers(min_value=1, max_value=10), min_size=4, max_size=4))
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("inplace", [True, False])
def test_sigmoid(capsys, ndim, dtype, inplace, shape):
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    sample = (10 * (2 * np.random.rand(*shape[:ndim]) - 1)).astype(dtype)
    input = wp.array(sample)
    output = warp_utils.sigmoid(input, inplace=inplace)
    assert output is input if inplace else output is not input
    assert np.allclose(output.numpy(), sigmoid(sample), atol=1e-05, rtol=1e-03)


@hypothesis.given(shape=st.lists(st.integers(min_value=1, max_value=10), min_size=4, max_size=4))
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("inplace", [True, False])
def test_softplus(capsys, ndim, dtype, inplace, shape):
    def softplus(x):
        return np.log(1.0 + np.exp(x))

    sample = (10 * (2 * np.random.rand(*shape[:ndim]) - 1)).astype(dtype)
    input = wp.array(sample)
    output = warp_utils.softplus(input, inplace=inplace)
    assert output is input if inplace else output is not input
    assert np.allclose(output.numpy(), softplus(sample), atol=1e-05, rtol=1e-03)


@hypothesis.given(shape=st.lists(st.integers(min_value=1, max_value=10), min_size=4, max_size=4))
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("inplace", [True, False])
def test_softsign(capsys, ndim, dtype, inplace, shape):
    def softsign(x):
        return x / (1.0 + np.abs(x))

    sample = (10 * (2 * np.random.rand(*shape[:ndim]) - 1)).astype(dtype)
    input = wp.array(sample)
    output = warp_utils.softsign(input, inplace=inplace)
    assert output is input if inplace else output is not input
    assert np.allclose(output.numpy(), softsign(sample), atol=1e-05, rtol=1e-03)


@hypothesis.given(shape=st.lists(st.integers(min_value=1, max_value=10), min_size=4, max_size=4))
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("inplace", [True, False])
def test_tanh(capsys, ndim, dtype, inplace, shape):
    def tanh(x):
        return np.tanh(x)

    sample = (10 * (2 * np.random.rand(*shape[:ndim]) - 1)).astype(dtype)
    input = wp.array(sample)
    output = warp_utils.tanh(input, inplace=inplace)
    assert output is input if inplace else output is not input
    assert np.allclose(output.numpy(), tanh(sample), atol=1e-05, rtol=1e-03)
