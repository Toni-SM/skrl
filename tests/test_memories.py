import string
import warnings
import hypothesis
import hypothesis.strategies as st
import pytest

import torch

from skrl.memories.torch import Memory, RandomMemory


@pytest.fixture
def classes_and_kwargs():
    return [(RandomMemory, {})]


@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_device(capsys, classes_and_kwargs, device):
    _device = torch.device(device) if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for klass, kwargs in classes_and_kwargs:
        try:
            memory: Memory = klass(memory_size=1, device=device, **kwargs)
        except (RuntimeError, AssertionError) as e:
            with capsys.disabled():
                print(e)
            warnings.warn(f"Invalid device: {device}. This test will be skipped")
            continue

        assert memory.device == _device  # defined device

@hypothesis.given(names=st.sets(st.text(alphabet=string.ascii_letters + string.digits + "_", min_size=1, max_size=10), min_size=1, max_size=10))
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
def test_create_tensors(capsys, classes_and_kwargs, names):
    for klass, kwargs in classes_and_kwargs:
        memory: Memory = klass(memory_size=1, **kwargs)

        for name in names:
            memory.create_tensor(name=name, size=1, dtype=torch.float32)

        assert memory.get_tensor_names() == sorted(names)

@hypothesis.given(memory_size=st.integers(min_value=1, max_value=100),
                  num_envs=st.integers(min_value=1, max_value=10),
                  num_samples=st.integers(min_value=1, max_value=500))
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
def test_add_samples(capsys, classes_and_kwargs, memory_size, num_envs, num_samples):
    for klass, kwargs in classes_and_kwargs:
        memory: Memory = klass(memory_size=memory_size, num_envs=num_envs, **kwargs)

        memory.create_tensor(name="tensor_1", size=1, dtype=torch.float32)
        memory.create_tensor(name="tensor_2", size=2, dtype=torch.float32)

        # memory_index
        for _ in range(num_samples):
            memory.add_samples(tensor_1=torch.zeros((num_envs, 1)))

        assert memory.memory_index == num_samples % memory_size
        assert memory.filled == (num_samples >= memory_size)

        memory.reset()

        # memory_index, env_index
        for _ in range(num_samples):
            memory.add_samples(tensor_2=torch.zeros((2,)))

        assert memory.memory_index == (num_samples // num_envs) % memory_size
        assert memory.env_index == num_samples % num_envs
        assert memory.filled == (num_samples >= memory_size * num_envs)
