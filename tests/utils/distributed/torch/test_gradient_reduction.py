from typing import Any

import pytest

from datetime import timedelta
from pathlib import Path
from gymnasium import spaces

import torch
import torch.multiprocessing as mp

from skrl import config
from skrl.models.torch import DeterministicMixin, Model


class _ChannelsLastPolicy(DeterministicMixin, Model):
    def __init__(self) -> None:
        Model.__init__(
            self,
            observation_space=spaces.Box(-10, 10, shape=(2, 5, 5)),
            action_space=spaces.Box(-10, 10, shape=(2,)),
            device="cpu",
        )
        DeterministicMixin.__init__(self)
        self.conv = torch.nn.Conv2d(2, 2, 3).to(dtype=torch.float64, memory_format=torch.channels_last)

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        return self.conv(inputs["observations"]).mean(dim=(-2, -1)), {}


def _backward(model: Model, rank: int) -> None:
    # element-wise and per-rank distinct inputs and output weights, so that every gradient element differs
    observations = torch.arange(2 * 2 * 5 * 5, dtype=torch.float64).reshape(2, 2, 5, 5).sin() + rank
    weights = torch.tensor([1.0, 2.0], dtype=torch.float64) * (rank + 1)
    model.zero_grad()
    (model.act({"observations": observations})[0] * weights).sum().backward()


def _reduce_channels_last_gradients(rank: int, world_size: int, rendezvous: str) -> None:
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    # skrl reads the world size from environment variables: set it explicitly for this spawned worker
    config.torch._world_size = world_size
    try:
        torch.manual_seed(0)  # same initial parameters for all workers
        model = _ChannelsLastPolicy()
        # expected gradients: average of the local gradients of all workers
        expected = []
        for i in range(world_size):
            _backward(model, i)
            expected.append([parameter.grad.clone() for parameter in model.parameters()])
        expected = [torch.stack(gradients).mean(dim=0) for gradients in zip(*expected)]
        # local gradients
        _backward(model, rank)
        assert not model.conv.weight.grad.is_contiguous()

        model.reduce_parameters()

        for parameter, gradient in zip(model.parameters(), expected):
            torch.testing.assert_close(parameter.grad, gradient)
        assert model.conv.weight.grad.is_contiguous(memory_format=torch.channels_last)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(), reason="Gloo is required"
)
def test_reduce_channels_last_gradients_on_two_cpu_workers(tmp_path: Path) -> None:
    mp.spawn(_reduce_channels_last_gradients, args=(2, (tmp_path / "rendezvous").as_uri()), nprocs=2, join=True)
