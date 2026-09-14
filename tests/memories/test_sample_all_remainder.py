import pytest

from importlib import import_module

import numpy as np


@pytest.mark.parametrize("framework", ["torch", "jax", "warp"])
@pytest.mark.parametrize("size, batches", [(10, 3), (11, 4), (12, 3), (2, 3)])
def test_sample_all_keeps_every_transition(framework: str, size: int, batches: int) -> None:
    lib = pytest.importorskip(framework)
    memory = import_module(f"skrl.memories.{framework}").RandomMemory(memory_size=size, device="cpu")
    memory.create_tensor("observations", size=1)
    values = np.arange(size, dtype=np.float32).reshape(-1, 1)
    for row in values:
        if framework == "torch":
            native = lib.tensor(row.reshape(1, 1))
        elif framework == "jax":
            native = lib.numpy.asarray(row.reshape(1, 1))
        else:
            native = lib.array(row.reshape(1, 1), device="cpu")
        memory.add_samples(observations=native)

    samples = memory.sample_all(["observations"], mini_batches=batches)

    assert len(samples) == batches
    arrays = [sample[0].numpy() if hasattr(sample[0], "numpy") else np.asarray(sample[0]) for sample in samples]
    np.testing.assert_array_equal(np.concatenate(arrays), values)
    assert max(len(x) for x in arrays) - min(len(x) for x in arrays) <= 1
