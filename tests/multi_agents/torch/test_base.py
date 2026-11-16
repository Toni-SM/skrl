import torch

from skrl.multi_agents.torch.base import ExperimentCfg, MultiAgent, MultiAgentCfg


class _TestMultiAgent(MultiAgent):
    def act(self, *args, **kwargs):
        raise NotImplementedError

    def pre_interaction(self, *args, **kwargs):
        raise NotImplementedError

    def post_interaction(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError


def test_record_transition_checks_finished_state_for_all_agents():
    """Check that a non-first agent can finish each vectorized environment."""
    possible_agents = ["agent_0", "agent_1", "agent_2"]
    agent = _TestMultiAgent(
        cfg=MultiAgentCfg(experiment=ExperimentCfg(write_interval=1)),
        possible_agents=possible_agents,
        models={uid: {} for uid in possible_agents},
    )

    agent.record_transition(
        observations={},
        states={},
        actions={},
        rewards={
            "agent_0": torch.tensor([[1.0], [2.0], [3.0]]),
            "agent_1": torch.tensor([[10.0], [20.0], [30.0]]),
            "agent_2": torch.zeros(3, 1),
        },
        next_observations={},
        next_states={},
        terminated={
            "agent_0": torch.zeros(3, 1, dtype=torch.bool),
            "agent_1": torch.tensor([[False], [True], [False]]),
            "agent_2": torch.zeros(3, 1, dtype=torch.bool),
        },
        truncated={
            "agent_0": torch.zeros(3, 1, dtype=torch.bool),
            "agent_1": torch.zeros(3, 1, dtype=torch.bool),
            "agent_2": torch.tensor([[False], [False], [True]]),
        },
        infos={},
        timestep=0,
        timesteps=1,
    )

    assert list(agent._track_rewards) == [[22.0], [33.0]]
    assert list(agent._track_timesteps) == [[1], [1]]
    torch.testing.assert_close(agent._cumulative_rewards, torch.tensor([[11.0], [0.0], [0.0]]))
    torch.testing.assert_close(agent._cumulative_timesteps, torch.tensor([[1], [0], [0]], dtype=torch.int32))
