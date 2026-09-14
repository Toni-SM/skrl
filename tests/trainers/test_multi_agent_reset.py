from typing import Any

import pytest

from importlib import import_module
import gymnasium

from tests.utilities import MultiAgentEnv, MultiAgentMock


class EndingMultiAgentEnv(MultiAgentEnv):
    def __init__(self, framework: str, agents_per_step: int) -> None:
        spaces = {uid: gymnasium.spaces.Box(-1, 1, (1,)) for uid in ("first", "second")}
        super().__init__(
            observation_spaces=spaces,
            state_spaces=spaces,
            action_spaces=spaces,
            num_envs=1,
            device="cpu",
            ml_framework=framework,
            probability=0,
        )
        self.agents_per_step = agents_per_step
        self.reset_count = 0
        self.remaining_agents = []

    def reset(self) -> Any:
        self.agents = self.possible_agents[:]
        self.num_agents = len(self.agents)
        self.reset_count += 1
        return super().reset()

    def step(self, actions: dict[str, Any]) -> Any:
        assert self.agents, "the trainer must reset an exhausted environment"
        observations, rewards, terminated, truncated, infos = super().step(actions)
        finished = self.agents[: self.agents_per_step]
        self.agents = self.agents[self.agents_per_step :]
        self.num_agents = len(self.agents)
        self.remaining_agents.append(self.num_agents)
        done = self._tensorize(True, bool)
        for uid in finished:
            terminated[uid] = done[uid]
        return observations, rewards, terminated, truncated, infos


@pytest.mark.parametrize("framework", ["torch", "jax", "warp"])
@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("agents_per_step", [1, 2])
def test_trainer_resets_when_the_last_agent_finishes(framework: str, mode: str, agents_per_step: int) -> None:
    pytest.importorskip(framework)
    module = import_module(f"skrl.trainers.{framework}.sequential")
    env = EndingMultiAgentEnv(framework, agents_per_step)
    agent = MultiAgentMock(
        possible_agents=env.possible_agents,
        observation_spaces=env.observation_spaces,
        state_spaces=env.state_spaces,
        action_spaces=env.action_spaces,
        num_envs=1,
        device="cpu",
        ml_framework=framework,
    )
    trainer = module.SequentialTrainer(
        cfg=module.SequentialTrainerCfg(
            timesteps=4, headless=True, disable_progressbar=True, close_environment_at_exit=False
        ),
        env=env,
        agents=agent,
    )
    getattr(trainer, mode)()

    assert env.remaining_agents == ([1, 0, 1, 0] if agents_per_step == 1 else [0, 0, 0, 0])
    assert env.reset_count == 1 + 2 * agents_per_step
