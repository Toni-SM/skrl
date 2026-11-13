import pytest

import torch

from skrl.memories.torch import RandomMemory


@pytest.mark.parametrize("initial_count", [0, 2, 4, 7])
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("num_fields", [1, 2])
def test_batched_insertion_matches_individual_transitions(initial_count: int, batch_size: int, num_fields: int) -> None:
    bulk = RandomMemory(memory_size=5, num_envs=1, device="cpu")
    reference = RandomMemory(memory_size=5, num_envs=1, device="cpu")
    names = ["observations", "actions"][:num_fields]
    for memory in (bulk, reference):
        for name in names:
            memory.create_tensor(name, size=1)
        for time in range(initial_count):
            memory.add_samples(**{name: torch.tensor([[100.0 * field + time]]) for field, name in enumerate(names)})

    values = {
        name: (torch.arange(initial_count, initial_count + batch_size).float() + 100 * field).reshape(-1, 1)
        for field, name in enumerate(names)
    }
    bulk.add_samples(**values)
    for index in range(batch_size):
        reference.add_samples(**{name: value[index : index + 1] for name, value in values.items()})

    for _ in range(2):
        assert len(bulk) == len(reference)
        assert bulk.memory_index == reference.memory_index
        assert bulk.filled == reference.filled
        for name in names:
            torch.testing.assert_close(
                bulk.get_tensor_by_name(name), reference.get_tensor_by_name(name), equal_nan=True
            )
        next_values = {name: value[-1:] + 1 for name, value in values.items()}
        bulk.add_samples(**next_values)
        reference.add_samples(**next_values)
