from typing import Any

import pytest

from importlib import import_module

import numpy as np


def _native(framework: str, data: Any) -> Any:
    module = import_module(framework)
    data = np.asarray(data, dtype=np.float32)
    if framework == "torch":
        return module.tensor(data, device="cpu")
    if framework == "jax":
        return module.numpy.asarray(data)
    return module.array(data, device="cpu")


def _numpy(data: Any) -> np.ndarray:
    return data.numpy() if hasattr(data, "numpy") else np.asarray(data)


@pytest.mark.parametrize("framework", ["torch", "jax", "warp"])
def test_creating_tensor_preserves_existing_transitions(framework: str) -> None:
    pytest.importorskip(framework)
    memory_type = import_module(f"skrl.memories.{framework}").RandomMemory
    memory = memory_type(memory_size=3, num_envs=1, device="cpu")
    memory.create_tensor("observations", size=1)
    for value in [1.0, 2.0]:
        memory.add_samples(observations=_native(framework, [[value]]))

    memory.create_tensor("rewards", size=1)

    assert len(memory) == 2
    sampled = memory.sample_by_index(["observations"], indexes=np.array([0, 1]))[0][0]
    np.testing.assert_array_equal(_numpy(sampled), [[1.0], [2.0]])
    assert np.isnan(_numpy(memory.get_tensor_by_name("rewards"))).all()


@pytest.mark.parametrize("framework", ["torch", "jax", "warp"])
@pytest.mark.parametrize(
    "initial_size, size, keep_dimensions, data_shape",
    [
        ((2, 3), [2, 3], True, (2, 3)),
        ([2, 3], (2, 3), True, (2, 3)),
        ((6,), 6, False, (6,)),
        (6, [6], True, (6,)),
        (6, 6, False, (6,)),
    ],
)
def test_recreating_tensor_preserves_existing_data(
    framework: str, initial_size: Any, size: Any, keep_dimensions: bool, data_shape: tuple[int, ...]
) -> None:
    pytest.importorskip(framework)
    memory_type = import_module(f"skrl.memories.{framework}").RandomMemory
    memory = memory_type(memory_size=3, num_envs=2, device="cpu")
    assert memory.create_tensor("hidden", size=initial_size, keep_dimensions=isinstance(initial_size, (tuple, list)))
    data = np.arange(36, dtype=np.float32).reshape(3, 2, *data_shape)
    memory.set_tensor_by_name("hidden", _native(framework, data))

    assert not memory.create_tensor("hidden", size=size, keep_dimensions=keep_dimensions)

    np.testing.assert_array_equal(_numpy(memory.get_tensor_by_name("hidden")), data)
    sampled = memory.sample_by_index(["hidden"], indexes=np.array([1, 4]))[0][0]
    np.testing.assert_array_equal(_numpy(sampled), data.reshape(6, *data_shape)[[1, 4]])


@pytest.mark.parametrize("framework", ["torch", "jax", "warp"])
@pytest.mark.parametrize("size, keep_dimensions", [((4, 3), True), ((3, 2), True), (3, False), (6, False)])
def test_recreating_tensor_rejects_different_shape(framework: str, size: Any, keep_dimensions: bool) -> None:
    pytest.importorskip(framework)
    memory_type = import_module(f"skrl.memories.{framework}").RandomMemory
    memory = memory_type(memory_size=3, num_envs=2, device="cpu")
    memory.create_tensor("hidden", size=(2, 3), keep_dimensions=True)

    with pytest.raises(ValueError, match="Tensor size"):
        memory.create_tensor("hidden", size=size, keep_dimensions=keep_dimensions)


@pytest.mark.parametrize("framework", ["torch", "jax", "warp"])
def test_recreating_tensor_rejects_different_dtype(framework: str) -> None:
    module = pytest.importorskip(framework)
    memory_type = import_module(f"skrl.memories.{framework}").RandomMemory
    memory = memory_type(memory_size=3, num_envs=2, device="cpu")
    memory.create_tensor("hidden", size=6)
    dtype = module.numpy.int32 if framework == "jax" else module.int32

    with pytest.raises(ValueError, match="Tensor dtype"):
        memory.create_tensor("hidden", size=6, dtype=dtype)
