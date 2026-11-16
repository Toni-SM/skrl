from typing import Any

import pytest

from gymnasium import spaces

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from skrl.models.jax import DeterministicMixin, Model


class ScalarPolicy(DeterministicMixin, Model):
    def __init__(self, **kwargs: Any) -> None:
        Model.__init__(self, **kwargs)
        DeterministicMixin.__init__(self)

    @nn.compact
    def __call__(self, inputs: dict[str, jax.Array], role: str = "") -> tuple[jax.Array, dict[str, Any]]:
        scale = self.param("scale", nn.initializers.constant(1.5), ())
        return scale * nn.Dense(2)(inputs["observations"]), {}


@pytest.fixture
def model() -> ScalarPolicy:
    policy = ScalarPolicy(
        observation_space=spaces.Box(low=-1, high=1, shape=(2,)),
        action_space=spaces.Box(low=-1, high=1, shape=(2,)),
        device="cpu",
    )
    policy.init_state_dict(key=jax.random.PRNGKey(17))
    return policy


def assert_same_tree(actual: Any, expected: Any) -> None:
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for result, original in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        assert result.shape == original.shape
        assert result.dtype == original.dtype
        np.testing.assert_array_equal(result, original)


def test_reduce_scalar_model_gradients(model: ScalarPolicy) -> None:
    """Reduce and apply gradients from a trainable scalar alongside dense parameters."""
    observations = jnp.array([[1.0, 2.0], [-1.0, 3.0]])

    def loss(params):
        actions, _ = model.act({"observations": observations}, params=params)
        return jnp.square(actions).mean()

    for _ in range(3):
        params = model.state_dict.params
        gradients = jax.grad(loss)(params)
        assert gradients["params"]["scale"].shape == ()
        reduced = model.reduce_parameters(gradients)
        assert_same_tree(reduced, gradients)
        updated = jax.tree_util.tree_map(lambda value, grad: value - 0.01 * grad, params, reduced)
        assert all(np.isfinite(value).all() for value in jax.tree_util.tree_leaves(updated))
        assert not np.array_equal(updated["params"]["scale"], params["params"]["scale"])
        model.state_dict = model.state_dict.replace(params=updated)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
def test_reduce_scalar_and_nested_mixed_shapes(model: ScalarPolicy, dtype: Any) -> None:
    scalar = jnp.array(2.5, dtype=dtype)
    assert_same_tree(model.reduce_parameters(scalar), scalar)
    tree = {
        "scalar": scalar,
        "nested": (jnp.arange(6, dtype=jnp.float32).reshape(2, 3), [jnp.array([1.0, -2.0])]),
        "empty": jnp.empty((0, 2), dtype=jnp.float32),
    }
    assert_same_tree(model.reduce_parameters(tree), tree)


def test_reduce_nonscalar_gradients(model: ScalarPolicy) -> None:
    tree = {"matrix": jnp.arange(6, dtype=jnp.float32).reshape(2, 3), "bias": jnp.array([2.0, -1.0, 4.0])}
    assert_same_tree(model.reduce_parameters(tree), tree)
