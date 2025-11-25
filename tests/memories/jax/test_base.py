import pytest

import jax
import jax.numpy as jnp
from skrl.memories.jax import RandomMemory


def test_sample_all():
    data_size = 5
    num_datapoints = 80
    mini_batches = 4
    memory = RandomMemory(memory_size=num_datapoints + 10, num_envs=1, replacement=True)
    memory.create_tensor(name="data", size=data_size)
    data = jax.random.normal(jax.random.PRNGKey(42), (num_datapoints, 1, data_size))
    for d in data:
        memory.add_samples(data=d)
    samples = memory.sample_all(["data"], mini_batches=mini_batches, sequence_length=1)
    samples = jnp.stack([s[0] for s in samples], axis=0)
    assert samples.shape == (mini_batches, num_datapoints // mini_batches, data_size)
    # Check that all datapoints are sampled
    for d in data:
        assert jnp.any(jnp.all(d == samples, axis=2), axis=(0, 1))
    # Check that all samples are from the dataset
    for s in samples.reshape(-1, data_size):
        assert jnp.any(jnp.all(s == data, axis=2), axis=(0, 1))
