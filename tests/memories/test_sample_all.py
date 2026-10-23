from typing import Any

import pytest

from importlib import import_module

import numpy as np


@pytest.mark.parametrize("framework", ["torch", "jax", "warp"])
@pytest.mark.parametrize("memory_size, num_envs, mini_batches", [(10, 1, 3), (5, 2, 3), (6, 2, 3), (2, 1, 5)])
def test_sample_all_retains_every_transition(
    framework: str, memory_size: int, num_envs: int, mini_batches: int
) -> None:
    module = pytest.importorskip(framework)
    memory = import_module(f"skrl.memories.{framework}").RandomMemory(
        memory_size=memory_size, num_envs=num_envs, device="cpu"
    )
    memory.create_tensor("observations", size=2)
    memory.create_tensor("hidden", size=(1, 2), keep_dimensions=True)
    expected = np.arange(memory_size * num_envs * 2, dtype=np.float32).reshape(memory_size, num_envs, 2)

    def native(data: np.ndarray) -> Any:
        if framework == "torch":
            return module.tensor(data, device="cpu")
        if framework == "jax":
            return module.numpy.asarray(data)
        return module.array(data, device="cpu")

    def numpy(data: Any) -> np.ndarray:
        return data.numpy() if hasattr(data, "numpy") else np.asarray(data)

    for observation in expected:
        memory.add_samples(observations=native(observation), hidden=native(observation[:, None, :] + 100))

    batches = memory.sample_all(["observations", "missing", "hidden"], mini_batches=mini_batches)

    assert len(batches) == mini_batches
    sizes = [batch[0].shape[0] for batch in batches]
    assert max(sizes) - min(sizes) <= 1
    np.testing.assert_array_equal(np.concatenate([numpy(batch[0]) for batch in batches]), expected.reshape(-1, 2))
    np.testing.assert_array_equal(
        np.concatenate([numpy(batch[2]) for batch in batches]), expected.reshape(-1, 1, 2) + 100
    )
    assert all(batch[1] is None for batch in batches)
    np.testing.assert_array_equal(numpy(memory.get_tensor_by_name("observations")), expected)
