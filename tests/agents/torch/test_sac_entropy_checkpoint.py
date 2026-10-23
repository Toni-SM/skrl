"""Resume learned SAC temperature together with its optimizer and models."""

from __future__ import annotations

from typing import Any

import pytest

import math
from pathlib import Path
import gymnasium

import torch

from skrl.agents.torch.sac import SAC, SAC_CFG, SAC_RNN
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


class _Policy(GaussianMixin, Model):
    def __init__(self, **kwargs: Any) -> None:
        Model.__init__(self, **kwargs)
        GaussianMixin.__init__(self)
        self.mean = torch.nn.Parameter(torch.tensor([0.2, -0.1]))
        self.log_std = torch.nn.Parameter(torch.tensor([-0.3, -0.4]))

    def compute(self, inputs: dict[str, torch.Tensor], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        return self.mean.expand(inputs["observations"].shape[0], -1), {"log_std": self.log_std}


class _Critic(DeterministicMixin, Model):
    def __init__(self, **kwargs: Any) -> None:
        Model.__init__(self, **kwargs)
        DeterministicMixin.__init__(self)
        self.weights = torch.nn.Parameter(torch.tensor([[0.1], [-0.2], [0.3], [0.4]]))
        self.bias = torch.nn.Parameter(torch.tensor([0.1]))

    def compute(self, inputs: dict[str, torch.Tensor], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        features = torch.cat((inputs["observations"], inputs["taken_actions"]), dim=-1)
        return features @ self.weights + self.bias, {}


def _agent(
    agent_class: type[SAC] | type[SAC_RNN],
    directory: Path,
    initial_entropy: float = 0.25,
    learn_entropy: bool = True,
) -> SAC | SAC_RNN:
    """Create real models and fill the actual replay memory."""
    observation_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(2,))
    action_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(2,))
    kwargs = {"observation_space": observation_space, "action_space": action_space, "device": "cpu"}
    models = {"policy": _Policy(**kwargs)}
    models.update({name: _Critic(**kwargs) for name in ("critic_1", "critic_2", "target_critic_1", "target_critic_2")})
    memory = RandomMemory(memory_size=4, num_envs=1, device="cpu")
    cfg = SAC_CFG(
        batch_size=2,
        gradient_steps=1,
        learning_rate=0.03,
        learn_entropy=learn_entropy,
        initial_entropy_value=initial_entropy,
        mixed_precision=False,
        experiment={"directory": str(directory), "write_interval": 0, "checkpoint_interval": 0},
    )
    agent = agent_class(models=models, memory=memory, **kwargs, cfg=cfg)
    agent.init()
    for index in range(4):
        memory.add_samples(
            observations=torch.tensor([[0.1 * index, -0.2]]),
            next_observations=torch.tensor([[0.1 * index + 0.05, -0.1]]),
            actions=torch.tensor([[0.2, -0.3]]),
            rewards=torch.tensor([[0.4 + 0.1 * index]]),
            terminated=torch.zeros((1, 1), dtype=torch.bool),
            truncated=torch.zeros((1, 1), dtype=torch.bool),
        )
    return agent


def _train_temperature(agent: SAC | SAC_RNN) -> None:
    """Perform complete public SAC updates that change the temperature."""
    for step in range(3):
        torch.manual_seed(41 + step)
        agent.update(timestep=step, timesteps=10)
    assert not math.isclose(float(agent.log_entropy_coefficient.exp()), agent.cfg.initial_entropy_value, rel_tol=1e-4)


@pytest.mark.parametrize("agent_class", [SAC, SAC_RNN])
def test_save_load_restores_learned_temperature_and_optimizer_parameter_identity(
    tmp_path: Path, agent_class: type[SAC] | type[SAC_RNN]
) -> None:
    """Restore the learned value without replacing the optimizer Parameter."""
    original = _agent(agent_class, tmp_path / "original")
    _train_temperature(original)
    path = tmp_path / "agent.pt"
    original.save(str(path))
    restored = _agent(agent_class, tmp_path / "restored", initial_entropy=0.7)
    parameter = restored.log_entropy_coefficient
    restored.load(str(path))
    assert restored.log_entropy_coefficient is parameter
    assert restored.entropy_optimizer.param_groups[0]["params"][0] is parameter
    torch.testing.assert_close(restored.log_entropy_coefficient, original.log_entropy_coefficient)
    assert float(restored._entropy_coefficient) == pytest.approx(float(original._entropy_coefficient))


@pytest.mark.parametrize("agent_class", [SAC, SAC_RNN])
def test_next_complete_sac_update_matches_uninterrupted_training(
    tmp_path: Path, agent_class: type[SAC] | type[SAC_RNN]
) -> None:
    """Match the next actor, critic and temperature update after resuming."""
    original = _agent(agent_class, tmp_path / "original")
    _train_temperature(original)
    path = tmp_path / "agent.pt"
    original.save(str(path))
    restored = _agent(agent_class, tmp_path / "restored")
    restored.load(str(path))
    for agent in (original, restored):
        torch.manual_seed(123)
        agent.update(timestep=3, timesteps=10)
    torch.testing.assert_close(restored.log_entropy_coefficient, original.log_entropy_coefficient)
    for name in original.models:
        for key, value in original.models[name].state_dict().items():
            torch.testing.assert_close(restored.models[name].state_dict()[key], value)


@pytest.mark.parametrize("agent_class", [SAC, SAC_RNN])
def test_legacy_checkpoint_without_temperature_uses_constructor_value(
    tmp_path: Path, agent_class: type[SAC] | type[SAC_RNN]
) -> None:
    """Use the constructor value when loading a legacy checkpoint."""
    original = _agent(agent_class, tmp_path / "original")
    path = tmp_path / "legacy.pt"
    original.save(str(path))
    checkpoint = torch.load(path, weights_only=True)
    checkpoint.pop("log_entropy_coefficient", None)
    torch.save(checkpoint, path)
    restored = _agent(agent_class, tmp_path / "restored", initial_entropy=0.7)
    restored.load(str(path))
    assert float(restored.log_entropy_coefficient.exp()) == pytest.approx(0.7)
    assert float(restored._entropy_coefficient) == pytest.approx(0.7)


@pytest.mark.parametrize("agent_class", [SAC, SAC_RNN])
def test_fixed_entropy_does_not_require_a_temperature_checkpoint_module(
    tmp_path: Path, agent_class: type[SAC] | type[SAC_RNN]
) -> None:
    """Preserve the fixed-entropy checkpoint behavior."""
    original = _agent(agent_class, tmp_path / "original", learn_entropy=False)
    assert "log_entropy_coefficient" not in original.checkpoint_modules
    path = tmp_path / "fixed.pt"
    original.save(str(path))
    restored = _agent(agent_class, tmp_path / "restored", learn_entropy=False)
    restored.load(str(path))
    assert restored._entropy_coefficient == original._entropy_coefficient == 0.25
