import pytest
import warnings
import hypothesis
import hypothesis.strategies as st

import torch

from skrl.trainers.torch import Trainer
from skrl.trainers.torch import ManualTrainer
from skrl.trainers.torch import ParallelTrainer
from skrl.trainers.torch import SequentialTrainer

from .utils import DummyEnv, DummyAgent


@pytest.fixture
def classes_and_kwargs():
    return [(ManualTrainer, {"cfg": {"timesteps": 100}}),
            (ParallelTrainer, {"cfg": {"timesteps": 100}}),
            (SequentialTrainer, {"cfg": {"timesteps": 100}})]


def test_train(capsys, classes_and_kwargs):
    env = DummyEnv(num_envs=1)
    agent = DummyAgent()

    for klass, kwargs in classes_and_kwargs:
        trainer: Trainer = klass(env, agents=agent, **kwargs)

        trainer.train()

def test_eval(capsys, classes_and_kwargs):
    env = DummyEnv(num_envs=1)
    agent = DummyAgent()

    for klass, kwargs in classes_and_kwargs:
        trainer: Trainer = klass(env, agents=agent, **kwargs)

        trainer.eval()
