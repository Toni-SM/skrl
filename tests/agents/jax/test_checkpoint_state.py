from typing import Any

import pytest

import pickle
from pathlib import Path
import gymnasium

import flax
import jax
import jax.numpy as jnp
import numpy as np

from skrl.agents.jax.cem import CEM
from skrl.memories.jax import RandomMemory
from skrl.multi_agents.jax.ippo import IPPO
from skrl.resources.preprocessors.jax import RunningStandardScaler
from skrl.utils.model_instantiators.jax import categorical_model, deterministic_model, gaussian_model


def assert_same_tree(actual: Any, expected: Any) -> None:
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for result, reference in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-7)


def make_cem(directory: Path, *, checkpoint_interval: int = 0, separated: bool = False) -> CEM:
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,))
    action_space = gymnasium.spaces.Discrete(2)
    policy = categorical_model(
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        network=[{"name": "net", "input": "OBSERVATIONS", "layers": [4], "activations": "tanh"}],
        output="ACTIONS",
    )
    policy.init_state_dict(key=jax.random.PRNGKey(17))
    agent = CEM(
        models={"policy": policy},
        memory=RandomMemory(memory_size=2, num_envs=1, device="cpu"),
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        cfg={
            "rollouts": 2,
            "discount_factor": 1.0,
            "percentile": 0.0,
            "learning_rate": 0.02,
            "experiment": {
                "directory": str(directory),
                "experiment_name": "checkpoint-test",
                "write_interval": 0,
                "checkpoint_interval": checkpoint_interval,
                "store_separately": separated,
            },
        },
    )
    agent.init()
    agent.enable_training_mode()
    return agent


def train_cem_episode(agent: CEM, episode: int) -> None:
    for offset in range(2):
        timestep = 2 * episode + offset
        observations = jnp.array([[0.25 * episode + offset]], dtype=jnp.float32)
        agent.record_transition(
            observations=observations,
            states=None,
            actions=jnp.array([[episode % 2]], dtype=jnp.int32),
            rewards=jnp.ones((1, 1)),
            next_observations=observations + 0.1,
            next_states=None,
            terminated=jnp.array([[offset == 1]]),
            truncated=jnp.zeros((1, 1), dtype=jnp.bool_),
            infos={},
            timestep=timestep,
            timesteps=8,
        )
        if offset == 1:
            agent.track_data("Reward / Total reward (mean)", 2.0)
        agent.post_interaction(timestep=timestep, timesteps=8)


def test_cem_checkpoint_restores_next_optimizer_update(tmp_path: Path) -> None:
    original = make_cem(tmp_path / "original")
    for episode in range(3):
        train_cem_episode(original, episode)
    path = tmp_path / "agent.pickle"
    original.save(str(path))
    restored = make_cem(tmp_path / "restored")
    restored.load(str(path))

    assert_same_tree(restored.policy.state_dict.params, original.policy.state_dict.params)
    assert_same_tree(restored.optimizer.state, original.optimizer.state)
    train_cem_episode(original, 3)
    train_cem_episode(restored, 3)
    assert_same_tree(restored.policy.state_dict.params, original.policy.state_dict.params)
    assert_same_tree(restored.optimizer.state, original.optimizer.state)


@pytest.mark.parametrize("separated", [False, True])
def test_periodic_and_best_checkpoints_capture_current_moments(tmp_path: Path, separated: bool) -> None:
    agent = make_cem(tmp_path, checkpoint_interval=2, separated=separated)
    train_cem_episode(agent, 0)
    best_state = agent.optimizer.state
    train_cem_episode(agent, 1)
    train_cem_episode(agent, 2)
    directory = Path(agent.experiment_dir) / "checkpoints"
    latest = pickle.loads((directory / ("optimizer_6.pickle" if separated else "agent_6.pickle")).read_bytes())
    best = pickle.loads((directory / ("best_optimizer.pickle" if separated else "best_agent.pickle")).read_bytes())
    if not separated:
        latest, best = latest["optimizer"], best["optimizer"]
    assert_same_tree(flax.serialization.from_bytes(agent.optimizer, latest).state, agent.optimizer.state)
    assert_same_tree(flax.serialization.from_bytes(agent.optimizer, best).state, best_state)


def test_model_only_checkpoint_keeps_new_optimizer_state(tmp_path: Path) -> None:
    original = make_cem(tmp_path / "original")
    train_cem_episode(original, 0)
    path = tmp_path / "legacy.pickle"
    original.save(str(path))
    modules = pickle.loads(path.read_bytes())
    modules.pop("optimizer")
    path.write_bytes(pickle.dumps(modules, protocol=4))
    restored = make_cem(tmp_path / "restored")
    initial_state = restored.optimizer.state
    restored.load(str(path))
    assert_same_tree(restored.policy.state_dict.params, original.policy.state_dict.params)
    assert_same_tree(restored.optimizer.state, initial_state)


def make_ippo(directory: Path) -> IPPO:
    possible_agents = ["first", "second"]
    observation_spaces = {uid: gymnasium.spaces.Box(low=-1, high=1, shape=(2,)) for uid in possible_agents}
    action_spaces = {uid: gymnasium.spaces.Box(low=-1, high=1, shape=(1,)) for uid in possible_agents}
    models = {}
    for index, uid in enumerate(possible_agents):
        models[uid] = {}
        for role, factory, output in [
            ("policy", gaussian_model, "ACTIONS"),
            ("value", deterministic_model, "ONE"),
        ]:
            model = factory(
                observation_space=observation_spaces[uid],
                action_space=action_spaces[uid],
                device="cpu",
                network=[{"name": "net", "input": "OBSERVATIONS", "layers": [4], "activations": "tanh"}],
                output=output,
            )
            model.init_state_dict(role=role, key=jax.random.PRNGKey(31 + index))
            models[uid][role] = model
    agent = IPPO(
        possible_agents=possible_agents,
        models=models,
        memories={uid: RandomMemory(memory_size=2, num_envs=2, device="cpu") for uid in possible_agents},
        observation_spaces=observation_spaces,
        state_spaces={uid: None for uid in possible_agents},
        action_spaces=action_spaces,
        device="cpu",
        cfg={
            "rollouts": 2,
            "learning_epochs": 2,
            "mini_batches": 1,
            "learning_rate": 0.002,
            "observation_preprocessor": RunningStandardScaler,
            "observation_preprocessor_kwargs": {"size": 2, "device": "cpu"},
            "experiment": {
                "directory": str(directory),
                "experiment_name": "checkpoint-test",
                "write_interval": 0,
                "checkpoint_interval": 0,
            },
        },
    )
    agent.init()
    agent.enable_training_mode()
    return agent


def test_ippo_checkpoint_restores_models_preprocessors_and_moments(tmp_path: Path) -> None:
    original = make_ippo(tmp_path / "original")
    for timestep in range(4):
        observations = {uid: jnp.array([[1.0, 2.0], [2.0, -1.0]]) + 0.1 * timestep for uid in original.possible_agents}
        states = {uid: None for uid in original.possible_agents}
        actions, _ = original.act(observations, states, timestep=timestep, timesteps=6)
        original.record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards={uid: jnp.array([[1.0], [-0.5]]) for uid in original.possible_agents},
            next_observations={uid: value + 0.1 for uid, value in observations.items()},
            next_states=states,
            terminated={uid: jnp.full((2, 1), timestep % 2 == 1) for uid in original.possible_agents},
            truncated={uid: jnp.zeros((2, 1), dtype=jnp.bool_) for uid in original.possible_agents},
            infos={},
            timestep=timestep,
            timesteps=6,
        )
        original.post_interaction(timestep=timestep, timesteps=6)
    path = tmp_path / "multi-agent.pickle"
    original.save(str(path))
    restored = make_ippo(tmp_path / "restored")
    restored.load(str(path))

    for uid in original.possible_agents:
        for role in ["policy", "value"]:
            assert_same_tree(restored.models[uid][role].state_dict.params, original.models[uid][role].state_dict.params)
            name = f"{role}_optimizer"
            assert_same_tree(getattr(restored, name)[uid].state, getattr(original, name)[uid].state)
        assert_same_tree(
            restored._observation_preprocessor[uid].state_dict.params,
            original._observation_preprocessor[uid].state_dict.params,
        )
        # Supply the same collected rollout; replay memory is not part of the agent checkpoint.
        for name in original.memories[uid].get_tensor_names():
            restored.memories[uid].set_tensor_by_name(name, original.memories[uid].get_tensor_by_name(name))
    restored._current_next_observations = original._current_next_observations.copy()
    restored._current_next_states = original._current_next_states.copy()
    for agent in [original, restored]:
        for uid in agent.possible_agents:
            agent.update(timestep=4, timesteps=6, uid=uid)
    for uid in original.possible_agents:
        for role in ["policy", "value"]:
            assert_same_tree(restored.models[uid][role].state_dict.params, original.models[uid][role].state_dict.params)
            name = f"{role}_optimizer"
            assert_same_tree(getattr(restored, name)[uid].state, getattr(original, name)[uid].state)
