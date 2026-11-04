from __future__ import annotations

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

    def compute(self, inputs: dict, role: str = "") -> tuple[torch.Tensor, dict]:
        return self.conv(inputs["observations"]).mean(dim=(-2, -1)), {}


def _reduce_channels_last_gradients(rank: int, world_size: int, rendezvous: str) -> None:
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    config.torch._world_size = world_size
    try:
        model = _ChannelsLastPolicy()
        observations = torch.full((2, 2, 5, 5), 2.0 * rank + 1, dtype=torch.float64)
        model.act({"observations": observations})[0].sum().backward()
        assert not model.conv.weight.grad.is_contiguous()

        model.reduce_parameters()

        torch.testing.assert_close(model.conv.weight.grad, torch.full_like(model.conv.weight, 4.0))
        torch.testing.assert_close(model.conv.bias.grad, torch.full_like(model.conv.bias, 2.0))
        assert model.conv.weight.grad.is_contiguous(memory_format=torch.channels_last)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is required")
def test_reduce_channels_last_gradients_on_two_cpu_workers(tmp_path: Path) -> None:
    mp.spawn(_reduce_channels_last_gradients, args=(2, (tmp_path / "rendezvous").as_uri()), nprocs=2, join=True)
