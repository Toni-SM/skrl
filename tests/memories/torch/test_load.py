import pytest

import torch

from skrl.memories.torch import RandomMemory


@pytest.mark.parametrize("format", ["pt", "npz"])
@pytest.mark.parametrize("num_envs", [1, 2])
def test_load_restores_sampled_tensors(tmp_path, format, num_envs):
    memory = RandomMemory(memory_size=3, num_envs=num_envs, device="cpu")
    memory.create_tensor(name="observations", size=2, dtype=torch.float32)
    memory.create_tensor(name="actions", size=1, dtype=torch.int64)
    observations = torch.arange(6 * num_envs, dtype=torch.float32).reshape(3, num_envs, 2)
    actions = torch.arange(3 * num_envs, dtype=torch.int64).reshape(3, num_envs, 1)
    for i in range(3):
        memory.add_samples(observations=observations[i], actions=actions[i])
    memory.save(str(tmp_path), format=format)
    path = next((tmp_path / "memories").glob(f"*.{format}"))
    memory.set_tensor_by_name("observations", torch.full_like(observations, -1))
    memory.set_tensor_by_name("actions", torch.full_like(actions, -1))

    memory.load(str(path))

    for name, expected in (("observations", observations), ("actions", actions)):
        actual = memory.get_tensor_by_name(name)
        torch.testing.assert_close(actual, expected)
        assert actual.dtype == expected.dtype
        assert getattr(memory, f"_tensor_{name}") is actual
        assert memory.tensors_view[name].data_ptr() == actual.data_ptr()
    sampled = memory.sample_by_index(["observations", "actions"], indexes=list(range(3 * num_envs)))[0]
    torch.testing.assert_close(sampled[0], observations.reshape(-1, 2))
    torch.testing.assert_close(sampled[1], actions.reshape(-1, 1))
    random_observations, random_actions = memory.sample(["observations", "actions"], batch_size=3 * num_envs)[0]
    torch.testing.assert_close(random_observations[:, :1], random_actions.float() * 2)

    # Subsequent writes must update the same storage used by sampling.
    memory.add_samples(observations=observations[0] + 20, actions=actions[0] + 10)
    sampled = memory.sample_by_index(["observations", "actions"], indexes=list(range(num_envs)))[0]
    torch.testing.assert_close(sampled[0], observations[0] + 20)
    torch.testing.assert_close(sampled[1], actions[0] + 10)


@pytest.mark.parametrize("format", ["pt", "npz"])
def test_load_preserves_multidimensional_data_shape(tmp_path, format):
    memory = RandomMemory(memory_size=3, num_envs=2, device="cpu")
    memory.create_tensor(name="hidden", size=(2, 3), keep_dimensions=True)
    expected = torch.arange(36, dtype=torch.float32).reshape(3, 2, 2, 3)
    memory.set_tensor_by_name("hidden", expected)
    memory.save(str(tmp_path), format=format)
    path = next((tmp_path / "memories").glob(f"*.{format}"))
    memory.set_tensor_by_name("hidden", torch.zeros_like(expected))

    memory.load(str(path))

    torch.testing.assert_close(memory.get_tensor_by_name("hidden"), expected)
    sampled = memory.sample_by_index(["hidden"], indexes=[1, 4])[0][0]
    torch.testing.assert_close(sampled, expected.reshape(6, 2, 3)[[1, 4]])


def test_load_numpy_registers_new_fields(tmp_path):
    source = RandomMemory(memory_size=3, num_envs=2, device="cpu")
    source.create_tensor(name="terminated", size=1, dtype=torch.bool)
    expected = torch.tensor([False, True, True, False, False, True]).reshape(3, 2, 1)
    source.set_tensor_by_name("terminated", expected)
    source.save(str(tmp_path), format="npz")
    path = next((tmp_path / "memories").glob("*.npz"))
    memory = RandomMemory(memory_size=3, num_envs=2, device="cpu")

    memory.load(str(path))

    assert memory.get_tensor_names() == ["terminated"]
    torch.testing.assert_close(memory.get_tensor_by_name("terminated"), expected)
    sampled = memory.sample_by_index(["terminated"], indexes=[1, 4])[0][0]
    torch.testing.assert_close(sampled, expected.reshape(6, 1)[[1, 4]])
    memory.tensors_view["terminated"][0] = True
    assert memory.get_tensor_by_name("terminated")[0, 0, 0]
    memory.get_tensor_by_name("terminated")[0, 0, 0] = False
    assert not memory.tensors_view["terminated"][0, 0]
