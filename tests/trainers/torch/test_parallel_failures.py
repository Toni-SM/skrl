import pytest

import json
import multiprocessing as mp
import threading
import time
from pathlib import Path
import gymnasium

from skrl.trainers.torch import ParallelTrainer

from ...utilities import AgentMock, SingleAgentEnv


class FailingAgent(AgentMock):
    def __init__(self, *, failure_stage, failure_path, **kwargs):
        super().__init__(**kwargs)
        self.failure_stage = failure_stage
        self.failure_path = failure_path

    def fail_if_requested(self, stage):
        if self.failure_stage == stage:
            Path(self.failure_path).write_text(stage)
            raise ValueError(f"parallel-regression-{stage}")

    def init(self, **kwargs):
        self.fail_if_requested("init")
        return super().init(**kwargs)

    def act(self, *args, **kwargs):
        self.fail_if_requested("act")
        return super().act(*args, **kwargs)

    def record_transition(self, **kwargs):
        self.fail_if_requested("record_transition")
        return super().record_transition(**kwargs)


class FailingEnvironment(SingleAgentEnv):
    def __init__(self, *, failure_path, **kwargs):
        super().__init__(**kwargs)
        self.failure_path = failure_path

    def step(self, actions):
        Path(self.failure_path).write_text("environment")
        raise ValueError("parallel-regression-environment")


def run_trainer_probe(mode, stage, failure_path, result_path):
    observation_space = gymnasium.spaces.Box(-1, 1, shape=(2,))
    action_space = gymnasium.spaces.Box(-1, 1, shape=(1,))
    env_kwargs = {
        "observation_space": observation_space,
        "state_space": None,
        "action_space": action_space,
        "num_envs": 2,
        "device": "cpu",
        "ml_framework": "torch",
        "probability": 0,
    }
    env = (
        FailingEnvironment(failure_path=failure_path, **env_kwargs)
        if stage == "environment"
        else SingleAgentEnv(**env_kwargs)
    )
    agents = [
        FailingAgent(
            observation_space=observation_space,
            state_space=None,
            action_space=action_space,
            num_envs=1,
            device="cpu",
            ml_framework="torch",
            failure_stage=stage if index == 0 else None,
            failure_path=failure_path,
        )
        for index in range(2)
    ]
    trainer = ParallelTrainer(
        env=env,
        agents=agents,
        scopes=[1, 1],
        cfg={
            "timesteps": 2,
            "headless": True,
            "disable_progressbar": True,
            "close_environment_at_exit": False,
        },
    )
    result = {}
    finished = threading.Event()

    def run():
        try:
            getattr(trainer, mode)()
        except BaseException as error:
            result.update(error=type(error).__name__, message=str(error))
        finally:
            finished.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 60
    if stage is not None:
        while not Path(failure_path).exists() and not finished.wait(0.05):
            if time.monotonic() >= deadline:
                break
    completed = finished.wait(10 if stage is not None else 60)
    children = mp.active_children()
    result.update(completed=completed, leaked_workers=len(children), fault_reached=Path(failure_path).exists())
    # Bound a regression on the unchanged implementation and avoid orphaning its workers.
    for process in children:
        process.terminate()
    for process in children:
        process.join()
    Path(result_path).write_text(json.dumps(result))


@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("stage", ["init", "act", "record_transition", "environment", None])
def test_parallel_trainer_reports_failures_and_reaps_workers(tmp_path, mode, stage):
    failure_path = tmp_path / "failure.txt"
    result_path = tmp_path / "result.json"
    process = mp.get_context("spawn").Process(
        target=run_trainer_probe, args=(mode, stage, str(failure_path), str(result_path))
    )
    process.start()
    try:
        process.join(timeout=90)
        assert not process.is_alive(), "The bounded parallel-trainer probe did not exit"
        assert process.exitcode == 0
        result = json.loads(result_path.read_text())
        assert result["completed"], f"Trainer hung after worker failure: {result}"
        assert result["leaked_workers"] == 0, result
        if stage is None:
            assert "error" not in result, result
        else:
            assert result["fault_reached"], result
            assert result["error"] == ("ValueError" if stage == "environment" else "RuntimeError"), result
            assert f"parallel-regression-{stage}" in result["message"]
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        process.close()
