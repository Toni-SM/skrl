"""The optional ``Agent.checkpoint_metadata`` attribute is carried through a grouped checkpoint.

A user may set ``agent.checkpoint_metadata`` to arbitrary, opaque metadata (run information, an
environment id, notes). When non-empty it is saved verbatim under the reserved ``"__metadata__"``
key in a grouped checkpoint and restored by ``load``. skrl takes no position on its contents.
"""
import dataclasses
import os
import tempfile

import gymnasium

from skrl.agents.torch.ppo import PPO as Agent
from skrl.agents.torch.ppo import PPO_CFG as AgentCfg
from skrl.memories.torch import RandomMemory
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model

from ...utilities import SingleAgentEnv


def _make_agent():
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,))
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(3,))
    env = SingleAgentEnv(
        observation_space=observation_space,
        state_space=None,
        action_space=action_space,
        num_envs=1,
        device="cpu",
        ml_framework="torch",
    )
    net = [{"name": "net", "input": "OBSERVATIONS", "layers": [4], "activations": "relu"}]
    models = {
        "policy": gaussian_model(
            observation_space=env.observation_space,
            state_space=env.state_space,
            action_space=env.action_space,
            device=env.device,
            network=net,
            output="ACTIONS",
        ),
        "value": deterministic_model(
            observation_space=env.observation_space,
            state_space=env.state_space,
            action_space=env.action_space,
            device=env.device,
            network=net,
            output="ONE",
        ),
    }
    memory = RandomMemory(memory_size=4, num_envs=env.num_envs, device=env.device)
    cfg = dataclasses.asdict(AgentCfg())
    cfg["experiment"]["write_interval"] = 0
    cfg["experiment"]["checkpoint_interval"] = 0
    return Agent(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
    )


def test_default_checkpoint_metadata_is_empty():
    assert _make_agent().checkpoint_metadata == {}


def test_metadata_saved_and_restored():
    meta = {"git_revision": "abc123", "env": "CartPole-v1", "note": "smoke run"}
    src = _make_agent()
    src.checkpoint_metadata = meta
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "agent.pt")
        src.save(path)

        dst = _make_agent()
        assert dst.checkpoint_metadata == {}
        dst.load(path)
        assert dst.checkpoint_metadata == meta


def test_no_metadata_key_when_unset():
    import torch

    src = _make_agent()  # checkpoint_metadata left empty
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "agent.pt")
        src.save(path)
        saved = torch.load(path, weights_only=False)
        assert "__metadata__" not in saved  # additive: no change to existing checkpoints

        # loading a metadata-less checkpoint leaves the attribute untouched and emits no warning
        dst = _make_agent()
        dst.load(path)
        assert dst.checkpoint_metadata == {}
