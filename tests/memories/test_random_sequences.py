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
@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize("num_envs,num_transitions", [(1, 3), (1, 7), (2, 6), (2, 14), (2, 13)])
def test_random_sequences_follow_retained_environment_history(
    framework: str, replacement: bool, num_envs: int, num_transitions: int
) -> None:
    pytest.importorskip(framework)
    memory_type = import_module(f"skrl.memories.{framework}").RandomMemory
    memory = memory_type(memory_size=5, num_envs=num_envs, device="cpu", replacement=replacement)
    memory.create_tensor("observations", size=2)
    memory.create_tensor("hidden", size=2)
    histories = [[] for _ in range(num_envs)]
    for index in range(num_transitions):
        time, env = divmod(index, num_envs)
        observation = np.array([time, env], dtype=np.float32)
        memory.add_samples(observations=_native(framework, observation), hidden=_native(framework, observation + 100))
        histories[env].append(time)
        histories[env] = histories[env][-5:]

    sampled, hidden = memory.sample(["observations", "hidden"], batch_size=20, sequence_length=3)[0]
    windows = _numpy(sampled).reshape(-1, 3, 2)
    actual = {tuple(window.reshape(-1)) for window in windows}
    expected = {
        tuple(value for time in times[start : start + 3] for value in (time, env))
        for env, times in enumerate(histories)
        for start in range(len(times) - 2)
    }

    assert actual
    assert actual <= expected
    if not replacement:
        assert actual == expected
        assert len(windows) == len(expected)
    np.testing.assert_array_equal(_numpy(hidden), _numpy(sampled) + 100)
    indexed = memory.sample_by_index(["observations"], indexes=memory.get_sampling_indexes())[0][0]
    np.testing.assert_array_equal(_numpy(indexed), _numpy(sampled))
