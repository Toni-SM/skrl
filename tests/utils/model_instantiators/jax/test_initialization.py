from typing import Any, Generator

import pytest

from gymnasium import spaces

import jax
import jax.numpy as jnp
import numpy as np

from skrl import config
from skrl.models.jax import Model
from skrl.utils.model_instantiators.jax import deterministic_model


@pytest.fixture(autouse=True)
def restore_key() -> Generator[None, Any, None]:
    key = config.jax.key
    try:
        yield
    finally:
        config.jax.key = key


def make_model() -> Model:
    return deterministic_model(
        observation_space=spaces.Box(low=-1, high=1, shape=(4,)),
        action_space=spaces.Box(low=-1, high=1, shape=(4,)),
        device="cpu",
        network=[{"name": "net", "input": "OBSERVATIONS", "layers": [4, 4], "activations": "tanh"}],
        output="ACTIONS",
    )


def assert_same_parameters(first: Any, second: Any) -> None:
    assert jax.tree_util.tree_structure(first) == jax.tree_util.tree_structure(second)
    for left, right in zip(jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second)):
        np.testing.assert_array_equal(left, right)


def test_default_initialization_advances_reproducibly() -> None:
    """Independent critics differ, while resetting the seed replays the sequence."""
    sequence = []
    observations = jnp.arange(24, dtype=jnp.float32).reshape(6, 4) / 24
    for _ in range(2):
        config.jax.key = 23
        models = [make_model(), make_model()]
        for model in models:
            model.init_state_dict({"observations": observations})
        first, _ = models[0].act({"observations": observations})
        second, _ = models[1].act({"observations": observations})
        assert not np.array_equal(first, second)
        sequence.append([model.state_dict.params for model in models])
    for first, replay in zip(*sequence):
        assert_same_parameters(first, replay)


def test_explicit_initialization_key_is_replayable() -> None:
    """An explicit key selects the same parameters without consuming global state."""
    original_key = np.asarray(config.jax.key)
    models = [make_model(), make_model()]
    for model in models:
        model.init_state_dict(key=jax.random.PRNGKey(41))
    assert_same_parameters(models[0].state_dict.params, models[1].state_dict.params)
    np.testing.assert_array_equal(config.jax.key, original_key)


@pytest.mark.parametrize(
    "method_name, parameter_name",
    [("init_parameters", None), ("init_weights", "kernel"), ("init_biases", "bias")],
)
def test_random_reinitialization_uses_independent_keys(method_name: str, parameter_name: str | None) -> None:
    """Same-shaped layers and consecutive initialization calls have distinct values."""
    model = make_model()
    model.init_state_dict(key=jax.random.PRNGKey(7))
    before = model.state_dict.params
    initialize = getattr(model, method_name)
    config.jax.key = 29
    initialize("normal", stddev=0.2)
    first = model.state_dict.params
    selected = []
    for (path, value), old in zip(jax.tree_util.tree_leaves_with_path(first), jax.tree_util.tree_leaves(before)):
        if parameter_name is None or path[-1].key == parameter_name:
            selected.append(value)
            assert not np.array_equal(value, old)
        else:
            np.testing.assert_array_equal(value, old)
    same_shape = [value for value in selected if value.shape == selected[0].shape]
    assert len(same_shape) > 1
    assert all(not np.array_equal(same_shape[0], value) for value in same_shape[1:])

    initialize("normal", stddev=0.2)
    second = model.state_dict.params
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second))
    )
    config.jax.key = 29
    initialize("normal", stddev=0.2)
    assert_same_parameters(first, model.state_dict.params)


def test_constant_initializers_preserve_parameter_selection() -> None:
    model = make_model()
    model.init_state_dict(key=jax.random.PRNGKey(13))
    model.init_weights("ones")
    model.init_biases("zeros")
    for path, value in jax.tree_util.tree_leaves_with_path(model.state_dict.params):
        expected = np.ones_like(value) if path[-1].key == "kernel" else np.zeros_like(value)
        np.testing.assert_array_equal(value, expected)
    model.init_parameters("ones")
    assert all(np.all(value == 1) for value in jax.tree_util.tree_leaves(model.state_dict.params))
