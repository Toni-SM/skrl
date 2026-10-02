import pytest


pytest.importorskip("jax")

from skrl.multi_agents.jax.mappo import MAPPO_CFG


def test_expand_empty_mapping():
    cfg = MAPPO_CFG(observation_preprocessor_kwargs={})
    cfg.expand(possible_agents=["agent_0", "agent_1"])
    assert cfg.observation_preprocessor_kwargs == {"agent_0": {}, "agent_1": {}}


def test_expand_agent_specific_mapping():
    cfg = MAPPO_CFG(observation_preprocessor_kwargs={"agent_0": {"epsilon": 0.1}, "agent_1": {}})
    cfg.expand(possible_agents=["agent_0", "agent_1"])
    assert cfg.observation_preprocessor_kwargs == {"agent_0": {"epsilon": 0.1}, "agent_1": {}}


def test_expand_missing_agents():
    cfg = MAPPO_CFG(observation_preprocessor_kwargs={"agent_0": {}})
    with pytest.raises(ValueError):
        cfg.expand(possible_agents=["agent_0", "agent_1"])
