import pytest

import jax
import numpy as np

from skrl import config


@pytest.fixture(autouse=True)
def restore_configuration():
    original_key = config.jax.key
    original_device = config.jax.device
    try:
        yield
    finally:
        config.jax.key = original_key
        config.jax.device = original_device


@pytest.mark.parametrize("implementation", ["threefry2x32", "rbg"])
def test_numpy_key_restores_the_complete_random_stream(implementation):
    with jax.default_prng_impl(implementation):
        expected_key = jax.random.split(jax.random.PRNGKey(23))[1]
        expected_samples = jax.random.normal(expected_key, (32,))
        config.jax.key = np.asarray(expected_key)
        np.testing.assert_array_equal(config.jax.key, expected_key)
        np.testing.assert_array_equal(jax.random.normal(config.jax.key, (32,)), expected_samples)


@pytest.mark.parametrize("typed", [False, True])
def test_device_assignment_preserves_an_existing_key(typed):
    key = jax.random.key(31) if typed else jax.random.PRNGKey(31)
    expected_key = jax.random.split(key)[1]
    expected_samples = jax.random.uniform(expected_key, (32,))
    config.jax.key = expected_key
    config.jax.device = "cpu"
    np.testing.assert_array_equal(jax.random.key_data(config.jax.key), jax.random.key_data(expected_key))
    np.testing.assert_array_equal(jax.random.uniform(config.jax.key, (32,)), expected_samples)
    assert config.jax.key.dtype == expected_key.dtype
    assert config.jax.key.device == jax.devices("cpu")[0]


@pytest.mark.parametrize("implementation", ["threefry2x32", "rbg"])
@pytest.mark.parametrize("seed", [0, 17, 17.9])
def test_numeric_seeds_use_the_configured_random_implementation(implementation, seed):
    with jax.default_prng_impl(implementation):
        config.jax.key = seed
        expected_key = jax.random.PRNGKey(np.asarray(seed, dtype=np.uint32))
        np.testing.assert_array_equal(config.jax.key, expected_key)
        np.testing.assert_array_equal(
            jax.random.normal(config.jax.key, (32,)),
            jax.random.normal(expected_key, (32,)),
        )


def test_explicit_jax_keys_remain_unchanged():
    key = jax.random.fold_in(jax.random.PRNGKey(41), 7)
    config.jax.key = key
    np.testing.assert_array_equal(config.jax.key, key)
    np.testing.assert_array_equal(jax.random.uniform(config.jax.key, (32,)), jax.random.uniform(key, (32,)))
