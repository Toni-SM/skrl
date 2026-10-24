import pytest

import dataclasses

from skrl.multi_agents.jax import MultiAgentCfg
from skrl.multi_agents.jax.ippo import IPPO_CFG
from skrl.multi_agents.jax.mappo import MAPPO_CFG


POSSIBLE_AGENTS = ["agent_0", "agent_1"]


@dataclasses.dataclass(kw_only=True)
class _Cfg(MultiAgentCfg):
    scalar: int = 1
    kwargs: dict = dataclasses.field(default_factory=dict)
    fixed: int = 0


@pytest.mark.parametrize("cfg_class", [IPPO_CFG, MAPPO_CFG])
def test_expand_default_config(capsys, cfg_class):
    cfg = cfg_class()
    cfg.expand(possible_agents=POSSIBLE_AGENTS)
    for uid in POSSIBLE_AGENTS:
        assert cfg.learning_rate_scheduler_kwargs[uid] == ({}, {})
        assert cfg.observation_preprocessor_kwargs[uid] == {}


def test_expand_shared_values(capsys):
    cfg = _Cfg(scalar=3, kwargs={"size": 4})
    cfg.expand(possible_agents=POSSIBLE_AGENTS)
    assert cfg.scalar == {uid: 3 for uid in POSSIBLE_AGENTS}
    assert cfg.kwargs == {uid: {"size": 4} for uid in POSSIBLE_AGENTS}
    # values are copied for each agent
    assert cfg.kwargs["agent_0"] is not cfg.kwargs["agent_1"]


def test_expand_per_agent_values(capsys):
    cfg = _Cfg(scalar={"agent_0": 1, "agent_1": 2}, kwargs={"agent_0": {"size": 4}, "agent_1": {"size": 3}})
    cfg.expand(possible_agents=POSSIBLE_AGENTS)
    assert cfg.scalar == {"agent_0": 1, "agent_1": 2}
    assert cfg.kwargs == {"agent_0": {"size": 4}, "agent_1": {"size": 3}}


@pytest.mark.parametrize("value", [{"agent_0": 1}, {"agent_0": 1, "size": 4}])
def test_expand_partial_per_agent_values(capsys, value):
    cfg = _Cfg(kwargs=value)
    with pytest.raises(ValueError):
        cfg.expand(possible_agents=POSSIBLE_AGENTS)


def test_expand_immutable(capsys):
    immutable = ["fixed"]
    cfg = _Cfg()
    cfg.expand(possible_agents=POSSIBLE_AGENTS, immutable=immutable)
    assert cfg.fixed == 0
    assert dataclasses.is_dataclass(cfg.experiment)
    # the caller's list is not modified
    assert immutable == ["fixed"]


def test_expand_twice(capsys):
    cfg = MAPPO_CFG(learning_rate=1e-3)
    cfg.expand(possible_agents=POSSIBLE_AGENTS)
    cfg.expand(possible_agents=POSSIBLE_AGENTS)
    assert cfg.learning_rate == {uid: (1e-3, 1e-3) for uid in POSSIBLE_AGENTS}
