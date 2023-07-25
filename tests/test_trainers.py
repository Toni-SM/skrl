import warnings
import hypothesis
import hypothesis.strategies as st
import pytest

import torch

from skrl.trainers.torch import ManualTrainer, ParallelTrainer, SequentialTrainer, Trainer

from .utils import DummyAgent, DummyEnv


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
