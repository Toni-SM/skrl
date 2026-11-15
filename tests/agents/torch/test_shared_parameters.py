from typing import Any

import pytest

import copy
import itertools
from pathlib import Path
import gymnasium

import torch

from skrl.agents.torch import Agent
from skrl.agents.torch.a2c import A2C, A2C_RNN
from skrl.agents.torch.ppo import PPO, PPO_RNN
from skrl.agents.torch.rpo import RPO, RPO_RNN
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


AGENTS = [A2C, A2C_RNN, PPO, PPO_RNN, RPO, RPO_RNN]
OBSERVATION_SPACE = gymnasium.spaces.Box(-1, 1, (3,))
ACTION_SPACE = gymnasium.spaces.Box(-1, 1, (2,))


def _features(model: Model, inputs: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    features = torch.tanh(model.encoder(inputs["observations"]))
    if model.rnn is None:
        return features, {}
    hidden = inputs["rnn"][0]
    if "terminated" in inputs:
        features = features.reshape(-1, 2, 4)
        hidden = hidden.reshape(1, -1, 2, 4)[:, :, 0].contiguous()
    else:
        features = features.unsqueeze(1)
    features, hidden = model.rnn(features, hidden)
    return features.reshape(-1, 4), {"rnn": [hidden]}


class _Policy(GaussianMixin, Model):
    def __init__(self, encoder: torch.nn.Linear, recurrent: bool) -> None:
        Model.__init__(self, observation_space=OBSERVATION_SPACE, action_space=ACTION_SPACE, device="cpu")
        GaussianMixin.__init__(self)
        self.encoder = encoder
        self.rnn = torch.nn.GRU(4, 4, batch_first=True) if recurrent else None
        self.head = torch.nn.Linear(4, 2)
        self.log_std = torch.nn.Parameter(torch.zeros(2))

    def get_specification(self) -> dict[str, Any]:
        return {"rnn": {"sequence_length": 2, "sizes": [(1, 2, 4)]}} if self.rnn is not None else {}

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        features, outputs = _features(self, inputs)
        mean = self.head(features)
        if "alpha" in inputs and "taken_actions" in inputs:
            mean = mean + torch.zeros_like(mean).uniform_(-inputs["alpha"], inputs["alpha"])
        return mean, {**outputs, "log_std": self.log_std}


class _Value(DeterministicMixin, Model):
    def __init__(self, encoder: torch.nn.Linear, recurrent: bool) -> None:
        Model.__init__(self, observation_space=OBSERVATION_SPACE, action_space=ACTION_SPACE, device="cpu")
        DeterministicMixin.__init__(self)
        self.encoder = encoder
        self.rnn = torch.nn.GRU(4, 4, batch_first=True) if recurrent else None
        self.head = torch.nn.Linear(4, 1)

    def get_specification(self) -> dict[str, Any]:
        return {"rnn": {"sequence_length": 2, "sizes": [(1, 2, 4)]}} if self.rnn is not None else {}

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        features, outputs = _features(self, inputs)
        return self.head(features), outputs


def _build_agent(agent_type: type[Agent], shared: bool = True, models: dict[str, Model] | None = None) -> Agent:
    if models is None:
        torch.manual_seed(42)
        recurrent = agent_type in (A2C_RNN, PPO_RNN, RPO_RNN)
        encoder = torch.nn.Linear(3, 4)
        models = {
            "policy": _Policy(encoder, recurrent),
            "value": _Value(encoder if shared else torch.nn.Linear(3, 4), recurrent),
        }
    cfg = {
        "rollouts": 2,
        "mini_batches": 1,
        "learning_rate": 1e-3,
        "grad_norm_clip": 0.05,
        "experiment": {"write_interval": 0, "checkpoint_interval": 0},
    }
    if agent_type not in (A2C, A2C_RNN):
        cfg["learning_epochs"] = 1
    agent = agent_type(
        models=models,
        memory=RandomMemory(memory_size=2, num_envs=2, device="cpu"),
        observation_space=OBSERVATION_SPACE,
        action_space=ACTION_SPACE,
        device="cpu",
        cfg=cfg,
    )
    agent.init()
    agent.enable_training_mode(True)
    return agent


def _parameters(agent: Agent) -> list[torch.nn.Parameter]:
    return list(torch.nn.ModuleList([agent.policy, agent.value]).parameters())


class _ReferenceAdam(torch.optim.Adam):
    def step(self, closure: Any = None) -> Any:
        parameters = self.param_groups[0]["params"]
        norm = torch.cat([parameter.grad.flatten() for parameter in parameters]).norm()
        scale = torch.clamp(0.05 / (norm + 1e-6), max=1.0)
        for parameter in parameters:
            parameter.grad.mul_(scale)
        return super().step(closure)


def _reference(agent: Agent) -> Agent:
    reference = _build_agent(type(agent), models=copy.deepcopy(agent.models))
    reference.optimizer = _ReferenceAdam(_parameters(reference), lr=1e-3)
    reference.checkpoint_modules["optimizer"] = reference.optimizer
    reference.cfg.grad_norm_clip = 0.0
    return reference


def _update(agent: Agent) -> None:
    torch.manual_seed(123)
    observations = torch.tensor([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]])
    with torch.no_grad():
        for step in range(2):
            actions, _ = agent.act(observations, None, timestep=step, timesteps=2)
            next_observations = observations + 0.1
            agent.record_transition(
                observations=observations,
                states=None,
                actions=actions,
                rewards=torch.tensor([[1.0 + step], [-2.0 + step]]),
                next_observations=next_observations,
                next_states=None,
                terminated=torch.zeros(2, 1, dtype=torch.bool),
                truncated=torch.zeros(2, 1, dtype=torch.bool),
                infos={},
                timestep=step,
                timesteps=2,
            )
            observations = next_observations
    agent.enable_models_training_mode(True)
    agent.update(timestep=1, timesteps=2)


def _assert_update_matches(actual: Agent, expected: Agent) -> None:
    for parameter, reference in zip(_parameters(actual), _parameters(expected)):
        torch.testing.assert_close(parameter.grad, reference.grad, atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(parameter, reference, atol=2e-7, rtol=2e-5)
        for name, state in actual.optimizer.state[parameter].items():
            torch.testing.assert_close(state, expected.optimizer.state[reference][name], atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("agent_type", AGENTS)
@pytest.mark.parametrize("shared", [True, False])
def test_update_counts_each_parameter_once(agent_type: type[Agent], shared: bool) -> None:
    agent = _build_agent(agent_type, shared=shared)
    reference = _reference(agent)
    _update(agent)
    _update(reference)
    _assert_update_matches(agent, reference)


@pytest.mark.parametrize("agent_type", AGENTS)
def test_legacy_shared_optimizer_checkpoint_resumes(agent_type: type[Agent], tmp_path: Path) -> None:
    legacy = _build_agent(agent_type)
    with pytest.warns(UserWarning, match="duplicate parameters"):
        legacy.optimizer = torch.optim.Adam(
            itertools.chain(legacy.policy.parameters(), legacy.value.parameters()), lr=1e-3
        )
    legacy.checkpoint_modules["optimizer"] = legacy.optimizer
    for step in range(2):
        for index, parameter in enumerate(_parameters(legacy)):
            parameter.grad = torch.full_like(parameter, (index + step + 1) / 10)
        legacy.optimizer.step()
    path = tmp_path / "legacy.pt"
    legacy.save(str(path))

    reference = _reference(legacy)
    for old, new in zip(_parameters(legacy), _parameters(reference)):
        reference.optimizer.state[new] = copy.deepcopy(legacy.optimizer.state[old])

    restored = _build_agent(agent_type)
    parameter_ids = [id(parameter) for parameter in _parameters(restored)]
    restored.load(str(path))
    assert parameter_ids == [id(parameter) for parameter in _parameters(restored)]
    for old, new in zip(_parameters(legacy), _parameters(restored)):
        for name, state in legacy.optimizer.state[old].items():
            torch.testing.assert_close(state, restored.optimizer.state[new][name])

    _update(restored)
    _update(reference)
    _assert_update_matches(restored, reference)


def test_distinct_parameter_checkpoint_is_not_deduplicated(tmp_path: Path) -> None:
    source = _build_agent(PPO, shared=False)
    path = tmp_path / "separate.pt"
    source.save(str(path))
    with pytest.raises(ValueError, match="parameter group"):
        _build_agent(PPO).load(str(path))


def test_shared_optimizer_checkpoint_roundtrip(tmp_path: Path) -> None:
    source = _build_agent(PPO)
    _update(source)
    path = tmp_path / "shared.pt"
    source.save(str(path))
    restored = _build_agent(PPO)
    restored.load(str(path))
    _update(source)
    _update(restored)
    _assert_update_matches(restored, source)
