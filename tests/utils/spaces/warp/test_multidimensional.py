from typing import Any

import pytest

from importlib import import_module
import gymnasium as gym

import numpy as np


@pytest.fixture(params=["warp"])
def backend(request: pytest.FixtureRequest) -> tuple[str, Any]:
    pytest.importorskip(request.param)
    return request.param, import_module(f"skrl.utils.spaces.{request.param}")


def _numpy(value: Any) -> np.ndarray:
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _space(shape: tuple[int, ...], container: str) -> gym.Space:
    discrete = gym.spaces.MultiDiscrete(np.arange(np.prod(shape)).reshape(shape) + 2)
    before = gym.spaces.Box(-2, 3, shape=(2,))
    after = gym.spaces.Box(-4, 5, shape=(1,))
    if container == "dict":
        return gym.spaces.Dict({"a": before, "b": discrete, "c": after})
    if container == "tuple":
        return gym.spaces.Tuple((before, discrete, after))
    return discrete


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 2, 3)])
@pytest.mark.parametrize("container", ["plain", "dict", "tuple"])
def test_multidimensional_size_and_limits(backend: tuple[str, Any], shape: tuple[int, ...], container: str) -> None:
    _, utils = backend
    space = _space(shape, container)
    count = int(np.prod(shape))
    width = count + (0 if container == "plain" else 3)
    assert utils.compute_space_size(space, occupied_size=True) == width
    assert utils.compute_space_size(space) == gym.spaces.flatdim(space)
    low, high = utils.compute_space_limits(space, occupied_size=True, device="cpu")
    expected_low = [-np.inf] * count
    expected_high = [np.inf] * count
    if container != "plain":
        expected_low = [-2, -2] + expected_low + [-4]
        expected_high = [3, 3] + expected_high + [5]
    np.testing.assert_array_equal(_numpy(low), expected_low)
    np.testing.assert_array_equal(_numpy(high), expected_high)


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 2, 3)])
@pytest.mark.parametrize("container", ["plain", "dict", "tuple"])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_multidimensional_round_trip(
    backend: tuple[str, Any], shape: tuple[int, ...], container: str, batch_size: int
) -> None:
    _, utils = backend
    space = _space(shape, container)
    original = utils.sample_space(space, batch_size=batch_size, backend="native", device="cpu")
    flat = utils.flatten_tensorized_space(original)
    assert flat.shape == (batch_size, int(np.prod(shape)) + (0 if container == "plain" else 3))
    restored = utils.unflatten_tensorized_space(space, flat)
    original_items = (
        original.values()
        if isinstance(original, dict)
        else original if isinstance(original, (tuple, list)) else [original]
    )
    restored_items = (
        restored.values()
        if isinstance(restored, dict)
        else restored if isinstance(restored, (tuple, list)) else [restored]
    )
    for expected, actual in zip(original_items, restored_items):
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(_numpy(actual), _numpy(expected))


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 2, 3)])
@pytest.mark.parametrize("container", ["plain", "dict", "tuple"])
def test_multidimensional_memory(backend: tuple[str, Any], shape: tuple[int, ...], container: str) -> None:
    framework, utils = backend
    space = _space(shape, container)
    memory_type = import_module(f"skrl.memories.{framework}").RandomMemory
    memory = memory_type(memory_size=2, num_envs=2, device="cpu")
    flat = utils.flatten_tensorized_space(utils.sample_space(space, batch_size=2, backend="native", device="cpu"))
    memory.create_tensor("observations", size=space, dtype=flat.dtype)
    memory.add_samples(observations=flat)
    assert memory.get_tensor_by_name("observations").shape == (2, 2, flat.shape[1])
    actual = memory.sample_by_index(["observations"], indexes=np.array([0, 1]))[0][0]
    np.testing.assert_array_equal(_numpy(actual), _numpy(flat))


def test_scalar_multidiscrete_size(backend: tuple[str, Any]) -> None:
    _, utils = backend
    space = gym.spaces.MultiDiscrete(np.array(5))
    size = utils.compute_space_size(space, occupied_size=True)
    assert isinstance(size, int)
    assert size == 1
    assert utils.compute_space_size(space) == 5
