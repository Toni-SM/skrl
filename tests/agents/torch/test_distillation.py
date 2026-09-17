import hypothesis
import hypothesis.strategies as st
import pytest

import copy
import dataclasses
import gymnasium

import torch

from skrl.agents.torch.distillation import DISTILLATION_CFG as AgentCfg
from skrl.agents.torch.distillation import Distillation as Agent
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.model_instantiators.torch import gaussian_model

from ...utilities import SingleAgentEnv, check_config_keys, get_test_mixed_precision, is_device_available


def _build_models(*, observation_space, state_space, action_space, device):
    """Instantiate a student (``policy``) and a privileged teacher model."""
    policy = gaussian_model(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device=device,
        network=[{"name": "net", "input": "OBSERVATIONS", "layers": [8], "activations": "elu"}],
        output="ACTIONS",
    )
    teacher = gaussian_model(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device=device,
        network=[
            {
                "name": "net",
                "input": "concatenate([OBSERVATIONS, STATES])" if state_space is not None else "OBSERVATIONS",
                "layers": [8],
                "activations": "elu",
            }
        ],
        output="ACTIONS",
    )
    policy.init_state_dict(role="policy")
    teacher.init_state_dict(role="teacher")
    return {"policy": policy, "teacher": teacher}


@hypothesis.given(
    num_envs=st.integers(min_value=1, max_value=5),
    # agent config
    rollouts=st.integers(min_value=1, max_value=5),
    learning_epochs=st.integers(min_value=1, max_value=5),
    mini_batches=st.integers(min_value=1, max_value=5),
    learning_rate=st.floats(min_value=1.0e-10, max_value=1),
    learning_rate_scheduler=st.one_of(st.none(), st.just(KLAdaptiveLR), st.just(torch.optim.lr_scheduler.ConstantLR)),
    learning_rate_scheduler_kwargs_value=st.floats(min_value=0.1, max_value=1),
    observation_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    state_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    grad_norm_clip=st.floats(min_value=0, max_value=1),
    loss_type=st.sampled_from(["mse", "huber"]),
    huber_delta=st.floats(min_value=0.1, max_value=2),
    mixed_precision=st.booleans(),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    max_examples=15,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("asymmetric", [True, False])
def test_agent(
    capsys,
    device,
    num_envs,
    asymmetric,
    # agent config
    rollouts,
    learning_epochs,
    mini_batches,
    learning_rate,
    learning_rate_scheduler,
    learning_rate_scheduler_kwargs_value,
    observation_preprocessor,
    state_preprocessor,
    grad_norm_clip,
    loss_type,
    huber_delta,
    mixed_precision,
):
    # check device availability
    if not is_device_available(device, backend="torch"):
        pytest.skip(f"Device {device} not available")

    # spaces
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,))
    state_space = gymnasium.spaces.Box(low=-1, high=1, shape=(5,)) if asymmetric else None
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(3,))

    # env
    env = SingleAgentEnv(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        num_envs=num_envs,
        device=device,
        ml_framework="torch",
    )

    # models
    models = _build_models(
        observation_space=observation_space, state_space=state_space, action_space=action_space, device=env.device
    )

    # memory
    memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=env.device)

    # agent
    cfg = {
        "rollouts": rollouts,
        "learning_epochs": learning_epochs,
        "mini_batches": mini_batches,
        "learning_rate": learning_rate,
        "learning_rate_scheduler": learning_rate_scheduler,
        "learning_rate_scheduler_kwargs": {},
        "observation_preprocessor": observation_preprocessor,
        "observation_preprocessor_kwargs": {"size": env.observation_space, "device": env.device},
        "state_preprocessor": state_preprocessor,
        "state_preprocessor_kwargs": {"size": env.state_space, "device": env.device},
        "grad_norm_clip": grad_norm_clip,
        "loss_type": loss_type,
        "huber_delta": huber_delta,
        "teacher_checkpoint": "",
        "mixed_precision": get_test_mixed_precision(mixed_precision),
        "experiment": {
            "directory": "",
            "experiment_name": "",
            "write_interval": 0,
            "checkpoint_interval": 0,
            "store_separately": False,
            "wandb": False,
            "wandb_kwargs": {},
        },
    }
    cfg["learning_rate_scheduler_kwargs"][
        "kl_threshold" if learning_rate_scheduler is KLAdaptiveLR else "factor"
    ] = learning_rate_scheduler_kwargs_value
    check_config_keys(cfg, dataclasses.asdict(AgentCfg()))
    check_config_keys(cfg["experiment"], dataclasses.asdict(AgentCfg().experiment))
    agent = Agent(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device=env.device,
    )

    # trainer
    cfg_trainer = {
        "timesteps": int(5 * rollouts),
        "headless": True,
        "disable_progressbar": True,
        "close_environment_at_exit": False,
    }
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    trainer.train()


def _build_trainer(*, rollouts=4, timesteps=8, asymmetric=True, device="cpu", num_envs=2, **agent_cfg):
    """Set up an environment, a memory and a distillation agent wired into a sequential trainer."""
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,))
    state_space = gymnasium.spaces.Box(low=-1, high=1, shape=(5,)) if asymmetric else None
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(3,))

    env = SingleAgentEnv(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        num_envs=num_envs,
        device=device,
        ml_framework="torch",
    )
    models = _build_models(
        observation_space=observation_space, state_space=state_space, action_space=action_space, device=env.device
    )
    memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=env.device)
    cfg = {"rollouts": rollouts, "experiment": {"write_interval": 0, "checkpoint_interval": 0}, **agent_cfg}
    agent = Agent(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device=env.device,
    )
    trainer = SequentialTrainer(
        cfg={
            "timesteps": timesteps,
            "headless": True,
            "disable_progressbar": True,
            "close_environment_at_exit": False,
        },
        env=env,
        agents=agent,
    )
    return trainer, agent, models


def test_update_optimizes_the_student_and_leaves_the_teacher_frozen():
    trainer, agent, models = _build_trainer()
    teacher_before = copy.deepcopy(models["teacher"].state_dict())
    policy_before = copy.deepcopy(models["policy"].state_dict())

    trainer.train()

    teacher_after = models["teacher"].state_dict()
    assert all(
        torch.equal(tensor, teacher_before[name]) for name, tensor in teacher_after.items()
    ), "the teacher's parameters must not change during training"
    policy_after = models["policy"].state_dict()
    assert any(
        not torch.equal(tensor, policy_before[name]) for name, tensor in policy_after.items()
    ), "the student's parameters must be updated during training"


def test_memory_stores_the_teacher_actions_as_regression_targets():
    """The stored targets must be the teacher's actions, not the student's."""
    _, agent, models = _build_trainer(rollouts=4, num_envs=2)
    agent.init()
    agent.enable_training_mode(True)

    observations = torch.rand((2, 4), device=agent.device)
    states = torch.rand((2, 5), device=agent.device)
    student_actions, _ = agent.act(observations, states, timestep=0, timesteps=8)
    agent.record_transition(
        observations=observations,
        states=states,
        actions=student_actions,
        rewards=torch.zeros((2, 1), device=agent.device),
        next_observations=observations,
        next_states=states,
        terminated=torch.zeros((2, 1), dtype=torch.bool, device=agent.device),
        truncated=torch.zeros((2, 1), dtype=torch.bool, device=agent.device),
        infos={},
        timestep=0,
        timesteps=8,
    )

    with torch.no_grad():
        _, outputs = models["teacher"].act({"observations": observations, "states": states}, role="teacher")
        expected = outputs["mean_actions"]
    stored = agent.memory.tensors["teacher_actions"][0]
    assert torch.allclose(stored, expected), "the stored targets must be the teacher's deterministic actions"
    assert not torch.allclose(stored, student_actions), "the stored targets must not be the student's sampled actions"


def test_update_consumes_every_transition_when_mini_batches_exceed_the_batch_size():
    """A mini-batch count larger than the number of transitions must not silently produce a no-op update."""
    trainer, agent, models = _build_trainer(rollouts=1, num_envs=1, timesteps=4, mini_batches=5)
    policy_before = copy.deepcopy(models["policy"].state_dict())

    trainer.train()

    losses = agent.tracking_data["Loss / Behavior cloning loss"]
    assert losses, "the update must have run"
    assert all(loss == loss for loss in losses), f"the behavior cloning loss must never be NaN, got {losses}"
    assert any(
        not torch.equal(tensor, policy_before[name]) for name, tensor in models["policy"].state_dict().items()
    ), "the student's parameters must be updated"


def test_update_on_an_empty_memory_warns_instead_of_crashing(caplog):
    _, agent, _ = _build_trainer()
    agent.init()

    agent.update(timestep=0, timesteps=8)

    assert "No transitions to update" in caplog.text
    assert "Loss / Behavior cloning loss" not in agent.tracking_data


def test_agent_raises_when_the_student_and_the_teacher_are_the_same_model():
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,))
    state_space = gymnasium.spaces.Box(low=-1, high=1, shape=(5,))
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(3,))

    models = _build_models(
        observation_space=observation_space, state_space=state_space, action_space=action_space, device="cpu"
    )
    models["teacher"] = models["policy"]

    with pytest.raises(ValueError, match="same"):
        Agent(
            models=models,
            memory=None,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )


def test_agent_trains_without_a_state_space():
    """Distilling a large teacher into a small student on identical observations is a valid use case."""
    trainer, agent, models = _build_trainer(asymmetric=False)
    policy_before = copy.deepcopy(models["policy"].state_dict())

    trainer.train()

    assert "states" not in agent.memory.tensors, "no state tensor should be allocated without a state space"
    assert any(
        not torch.equal(tensor, policy_before[name]) for name, tensor in models["policy"].state_dict().items()
    ), "the student's parameters must be updated during training"


def test_checkpoint_round_trip_restores_the_student_and_the_teacher(tmp_path):
    _, agent, models = _build_trainer()
    path = tmp_path / "agent.pt"
    agent.save(str(path))

    _, restored_agent, restored_models = _build_trainer()
    restored_agent.load(str(path))

    for role in ["policy", "teacher"]:
        for name, tensor in models[role].state_dict().items():
            assert torch.equal(tensor, restored_models[role].state_dict()[name]), f"'{role}.{name}' was not restored"


def test_runner_instantiates_the_agent_from_a_configuration():
    """The agent must be reachable through the runner, including the privileged teacher's input expression."""
    from skrl.utils.runner.torch import Runner

    env = SingleAgentEnv(
        observation_space=gymnasium.spaces.Box(low=-1, high=1, shape=(4,)),
        state_space=gymnasium.spaces.Box(low=-1, high=1, shape=(5,)),
        action_space=gymnasium.spaces.Box(low=-1, high=1, shape=(3,)),
        num_envs=2,
        device="cpu",
        ml_framework="torch",
    )
    network = [{"name": "net", "input": "OBSERVATIONS", "layers": [8], "activations": "elu"}]
    teacher_network = [
        {"name": "net", "input": "concatenate([OBSERVATIONS, STATES])", "layers": [8], "activations": "elu"}
    ]
    cfg = {
        "models": {
            "separate": True,
            "policy": {"class": "GaussianMixin", "network": network, "output": "ACTIONS"},
            "teacher": {"class": "GaussianMixin", "network": teacher_network, "output": "ACTIONS"},
        },
        "memory": {"class": "RandomMemory", "memory_size": -1},
        "agent": {"class": "Distillation", "rollouts": 4},
        "trainer": {"class": "SequentialTrainer", "timesteps": 8, "close_environment_at_exit": False},
    }

    runner = Runner(env, cfg)

    assert isinstance(runner.agent, Agent)
    assert runner.agent.policy is runner.agent.models["policy"]
    assert runner.agent.teacher is runner.agent.models["teacher"]
    runner.trainer.train()


def _perturbed_teacher_state_dict(models):
    """Return a copy of the teacher's state dict with every floating-point tensor altered."""
    state_dict = copy.deepcopy(models["teacher"].state_dict())
    for name, tensor in state_dict.items():
        if torch.is_floating_point(tensor):
            state_dict[name] = tensor + 1.0
    return state_dict


def _assert_teacher_matches(models, state_dict):
    for name, tensor in models["teacher"].state_dict().items():
        assert torch.equal(tensor, state_dict[name]), f"teacher parameter '{name}' was not loaded"


def test_load_teacher_from_a_whole_agent_checkpoint(tmp_path):
    _, agent, models = _build_trainer()
    expected = _perturbed_teacher_state_dict(models)
    path = tmp_path / "agent_1000.pt"
    torch.save({"policy": expected, "optimizer": {}}, path)

    agent.load_teacher(str(path))

    _assert_teacher_matches(models, expected)


def test_load_teacher_prefers_the_teacher_entry_over_the_policy_entry(tmp_path):
    _, agent, models = _build_trainer()
    expected = _perturbed_teacher_state_dict(models)
    other = {name: tensor * -3.0 for name, tensor in expected.items()}
    path = tmp_path / "agent_1000.pt"
    torch.save({"policy": other, "teacher": expected}, path)

    agent.load_teacher(str(path))

    _assert_teacher_matches(models, expected)


def test_load_teacher_from_a_single_model_state_dict(tmp_path):
    _, agent, models = _build_trainer()
    expected = _perturbed_teacher_state_dict(models)
    path = tmp_path / "policy_1000.pt"
    torch.save(expected, path)

    agent.load_teacher(str(path))

    _assert_teacher_matches(models, expected)


def test_load_teacher_raises_when_no_usable_state_dict_is_found(tmp_path):
    _, agent, _ = _build_trainer()
    path = tmp_path / "agent_1000.pt"
    torch.save({"optimizer": {"state": {}}, "value": {"weight": {}}}, path)

    with pytest.raises(ValueError, match="optimizer"):
        agent.load_teacher(str(path))


def test_load_teacher_warns_about_preprocessors_it_cannot_apply(tmp_path, caplog):
    """The teacher's normalization lives in the agent that trained it, not in the model's state dict."""
    _, agent, models = _build_trainer()
    path = tmp_path / "agent_1000.pt"
    torch.save({"policy": _perturbed_teacher_state_dict(models), "observation_preprocessor": {}}, path)

    agent.load_teacher(str(path))

    assert "observation_preprocessor" in caplog.text
    assert "preprocess" in caplog.text


def test_loading_a_whole_agent_checkpoint_marks_the_teacher_as_loaded(tmp_path, caplog):
    _, agent, _ = _build_trainer()
    path = tmp_path / "agent.pt"
    agent.save(str(path))

    _, restored_agent, _ = _build_trainer()
    restored_agent.load(str(path))
    caplog.clear()
    restored_agent.init()

    assert "have not been loaded" not in caplog.text, "restoring a checkpoint must not warn about an unloaded teacher"


def test_load_teacher_keeps_the_teacher_frozen(tmp_path):
    _, agent, models = _build_trainer()
    path = tmp_path / "policy_1000.pt"
    torch.save(_perturbed_teacher_state_dict(models), path)

    agent.load_teacher(str(path))

    assert not any(parameter.requires_grad for parameter in models["teacher"].parameters())


def test_teacher_checkpoint_config_loads_the_teacher_at_construction(tmp_path):
    _, _, models = _build_trainer()
    expected = _perturbed_teacher_state_dict(models)
    path = tmp_path / "policy_1000.pt"
    torch.save(expected, path)

    _, _, loaded_models = _build_trainer(teacher_checkpoint=str(path))

    _assert_teacher_matches(loaded_models, expected)


def test_agent_raises_when_policy_model_is_missing():
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,))
    state_space = gymnasium.spaces.Box(low=-1, high=1, shape=(5,))
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(3,))

    models = _build_models(
        observation_space=observation_space, state_space=state_space, action_space=action_space, device="cpu"
    )
    # the student must be given under the 'policy' role: 'student' is not an accepted key
    models["student"] = models.pop("policy")

    with pytest.raises(ValueError, match="policy"):
        Agent(
            models=models,
            memory=None,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )


def test_agent_raises_when_teacher_model_is_missing():
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,))
    state_space = gymnasium.spaces.Box(low=-1, high=1, shape=(5,))
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(3,))

    models = _build_models(
        observation_space=observation_space, state_space=state_space, action_space=action_space, device="cpu"
    )
    del models["teacher"]

    with pytest.raises(ValueError, match="teacher"):
        Agent(
            models=models,
            memory=None,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )


def test_agent_raises_for_discrete_action_spaces():
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,))
    state_space = gymnasium.spaces.Box(low=-1, high=1, shape=(5,))
    action_space = gymnasium.spaces.Discrete(3)

    models = _build_models(
        observation_space=observation_space, state_space=state_space, action_space=action_space, device="cpu"
    )

    with pytest.raises(ValueError, match="continuous"):
        Agent(
            models=models,
            memory=None,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device="cpu",
        )


def test_config_raises_for_unknown_loss_type():
    with pytest.raises(ValueError, match="loss_type"):
        AgentCfg(loss_type="l1").validate()
