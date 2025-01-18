import hypothesis
import hypothesis.strategies as st
import pytest

import torch

from skrl import config
from skrl.memories.torch import Memory


@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_device(capsys, device):
    memory = Memory(memory_size=5, num_envs=1, device=device)
    memory.create_tensor("buffer", size=1)

    target_device = config.torch.parse_device(device)
    assert memory.device == target_device
    assert memory.get_tensor_by_name("buffer").device == target_device


# __len__


def test_share_memory(capsys):
    memory = Memory(memory_size=5, num_envs=1, device="cuda")
    memory.create_tensor("buffer", size=1)

    memory.share_memory()


@hypothesis.given(
    tensor_names=st.lists(
        st.text(st.characters(codec="ascii", categories=("Nd", "L")), min_size=1, max_size=5),  # codespell:ignore
        min_size=0,
        max_size=5,
        unique=True,
    )
)
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
def test_get_tensor_names(capsys, tensor_names):
    memory = Memory(memory_size=5, num_envs=1)
    for name in tensor_names:
        memory.create_tensor(name, size=1)

    assert memory.get_tensor_names() == sorted(tensor_names)


@hypothesis.given(
    tensor_name=st.text(
        st.characters(codec="ascii", categories=("Nd", "L")), min_size=1, max_size=5  # codespell:ignore
    )
)
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
@pytest.mark.parametrize("keepdim", [True, False])
def test_get_tensor_by_name(capsys, tensor_name, keepdim):
    memory = Memory(memory_size=5, num_envs=2)
    memory.create_tensor(tensor_name, size=1)

    target_shape = (5, 2, 1) if keepdim else (10, 1)
    assert memory.get_tensor_by_name(tensor_name, keepdim=keepdim).shape == target_shape


@hypothesis.given(
    tensor_name=st.text(
        st.characters(codec="ascii", categories=("Nd", "L")), min_size=1, max_size=5  # codespell:ignore
    )
)
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
def test_set_tensor_by_name(capsys, tensor_name):
    memory = Memory(memory_size=5, num_envs=2)
    memory.create_tensor(tensor_name, size=1)

    target_tensor = torch.arange(10, device=memory.device).reshape(5, 2, 1)
    memory.set_tensor_by_name(tensor_name, target_tensor)
    assert torch.any(memory.get_tensor_by_name(tensor_name, keepdim=True) == target_tensor)
