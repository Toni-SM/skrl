import warnings
import hypothesis
import hypothesis.strategies as st
import pytest

import torch

from skrl.agents.torch import Agent
from skrl.agents.torch.a2c import A2C
from skrl.agents.torch.amp import AMP
from skrl.agents.torch.cem import CEM
from skrl.agents.torch.ddpg import DDPG
from skrl.agents.torch.dqn import DDQN, DQN
from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.q_learning import Q_LEARNING
from skrl.agents.torch.sac import SAC
from skrl.agents.torch.sarsa import SARSA
from skrl.agents.torch.td3 import TD3
from skrl.agents.torch.trpo import TRPO

from .utils import DummyModel


@pytest.fixture
def classes_and_kwargs():
    return [(A2C, {"models": {"policy": DummyModel()}}),
            (AMP, {"models": {"policy": DummyModel()}}),
            (CEM, {"models": {"policy": DummyModel()}}),
            (DDPG, {"models": {"policy": DummyModel()}}),
            (DQN, {"models": {"policy": DummyModel()}}),
            (DDQN, {"models": {"policy": DummyModel()}}),
            (PPO, {"models": {"policy": DummyModel()}}),
            (Q_LEARNING, {"models": {"policy": DummyModel()}}),
            (SAC, {"models": {"policy": DummyModel()}}),
            (SARSA, {"models": {"policy": DummyModel()}}),
            (TD3, {"models": {"policy": DummyModel()}}),
            (TRPO, {"models": {"policy": DummyModel()}})]


def test_agent(capsys, classes_and_kwargs):
    for klass, kwargs in classes_and_kwargs:
        cfg = {"learning_starts": 1,
               "experiment": {"write_interval": 0}}
        agent: Agent = klass(cfg=cfg, **kwargs)

        agent.init()
        agent.pre_interaction(timestep=0, timesteps=1)
        # agent.act(None, timestep=0, timestesps=1)
        agent.record_transition(states=torch.tensor([]),
                                actions=torch.tensor([]),
                                rewards=torch.tensor([]),
                                next_states=torch.tensor([]),
                                terminated=torch.tensor([]),
                                truncated=torch.tensor([]),
                                infos={},
                                timestep=0,
                                timesteps=1)
        agent.post_interaction(timestep=0, timesteps=1)
