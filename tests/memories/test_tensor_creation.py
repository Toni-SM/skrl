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
