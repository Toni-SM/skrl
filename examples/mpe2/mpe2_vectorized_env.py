from typing import Any, Callable

import multiprocessing

import gymnasium
import numpy as np

from pettingzoo import ParallelEnv


class _EnvGroup:
    def __init__(self, env_fn: Callable[[int], ParallelEnv], indexes: list[int]) -> None:
        """Group of environment instances stepped sequentially (in the main process or in a worker process).

        :param env_fn: Function that creates an environment instance, given its index.
        :param indexes: Indexes of the environment instances of the group.
        """
        self.envs = [env_fn(i) for i in indexes]
        self.indexes = indexes
        self.options = None

    def spaces(self) -> dict[str, Any]:
        """Get the agents and spaces (and other attributes) of the first environment instance."""
        env = self.envs[0]
        return {
            "metadata": env.metadata,
            "render_mode": env.render_mode,
            "possible_agents": env.possible_agents[:],
            "observation_spaces": {uid: env.observation_space(uid) for uid in env.possible_agents},
            "action_spaces": {uid: env.action_space(uid) for uid in env.possible_agents},
            "state_space": env.state_space,
        }

    def reset(self, seed: int | None, options: dict | None) -> list[tuple[dict, dict]]:
        """Reset the environment instances (each one is seeded with ``seed`` plus its index)."""
        self.options = options
        return [
            env.reset(seed=None if seed is None else seed + index, options=options)
            for index, env in zip(self.indexes, self.envs)
        ]

    def step(self, actions: list[dict[str, Any]]) -> list[tuple]:
        """Step the environment instances and auto-reset those whose agents are all done.

        :return: Observations, rewards, terminations, truncations, info, final observations (the observations
            with which the transitions ended: they differ from the observations if the environment instance was reset)
            and final state, for each environment instance.
        """
        outputs = []
        for env, env_actions in zip(self.envs, actions):
            observations, rewards, terminations, truncations, info = env.step(env_actions)
            final_observations, final_state = observations, env.state()
            if not env.agents:
                observations, _ = env.reset(options=self.options)
            outputs.append((observations, rewards, terminations, truncations, info, final_observations, final_state))
        return outputs

    def state(self) -> list[np.ndarray]:
        """Get the state of the environment instances."""
        return [env.state() for env in self.envs]

    def render(self) -> Any:
        """Render the first environment instance."""
        return self.envs[0].render()

    def close(self) -> None:
        """Close the environment instances."""
        for env in self.envs:
            env.close()


def _worker(remote: Any, parent_remote: Any, env_fn: Callable[[int], ParallelEnv], indexes: list[int]) -> None:
    """Worker process loop: execute the group's methods requested by the main process.

    Each response is a ``(success, result)`` tuple, where the result is the raised exception if the method failed.
    """
    parent_remote.close()
    try:
        group = _EnvGroup(env_fn, indexes)
        remote.send((True, None))
    except Exception as e:
        remote.send((False, e))
        remote.close()
        return
    try:
        while True:
            command, args = remote.recv()
            try:
                remote.send((True, getattr(group, command)(*args)))
            except Exception as e:
                remote.send((False, e))
            if command == "close":
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        remote.close()


class VectorizedParallelEnv(ParallelEnv):
    def __init__(self, env_fn: Callable[[int], ParallelEnv], num_envs: int, num_workers: int = 0) -> None:
        """Vectorized PettingZoo (Parallel API) environment whose outputs are batched per agent.

        As in Isaac Lab, each agent's observations, rewards, terminations and truncations (and the environment state)
        are returned as arrays whose first dimension is the number of environments, and the environments are
        auto-reset (at the same step) when all their agents are done. Therefore, the observations and state returned
        by the step of an environment that was reset are the initial ones of its new episode. The observations and
        states with which the transitions ended (the final ones, for the environments that were reset) are available
        in the step's info, as batches, under the ``"final_observations"`` (a dictionary mapping each agent to its
        batched observations) and ``"final_states"`` keys. skrl's multi-agents use them to bootstrap at truncation.
        The step's and reset's infos are not indexed by agent: the per-environment infos are listed under ``"envs"``.

        The environment instances are split (in contiguous groups) between ``num_workers`` worker processes,
        which step them in parallel, or are stepped sequentially in the main process if ``num_workers`` is 0.
        Worker processes are started with the ``fork`` method (only available on POSIX systems, e.g.: Linux),
        so ``env_fn`` can be any callable (e.g.: a lambda function).

        The agents must be alive during the whole episode (e.g.: MPE2 environments).

        :param env_fn: Function that creates an environment instance, given its index.
            Only the first environment instance (index 0) is rendered.
        :param num_envs: Number of environment instances.
        :param num_workers: Number of worker processes (at most ``num_envs`` are started).
            If 0, the environment instances are stepped in the main process.

        :raises ValueError: If the number of environment instances or worker processes is not valid,
            or if worker processes are requested and the ``fork`` start method is not available.
        """
        if num_envs < 1:
            raise ValueError(f"The number of environments must be greater than 0, got {num_envs}")
        if num_workers < 0:
            raise ValueError(f"The number of worker processes must be greater than or equal to 0, got {num_workers}")
        if num_workers and "fork" not in multiprocessing.get_all_start_methods():
            raise ValueError("Worker processes require the 'fork' start method (not available on this platform)")

        self.num_envs = num_envs
        self._groups = [
            group.tolist() for group in np.array_split(np.arange(num_envs), max(1, min(num_workers, num_envs)))
        ]
        self._remotes, self._processes, self._local = [], [], None
        self._closed = False
        if num_workers:
            context = multiprocessing.get_context("fork")
            for indexes in self._groups:
                remote, work_remote = context.Pipe()
                process = context.Process(target=_worker, args=(work_remote, remote, env_fn, indexes), daemon=True)
                process.start()
                work_remote.close()
                self._remotes.append(remote)
                self._processes.append(process)
            # wait for the environment instances to be created
            self._receive(self._remotes)
        else:
            self._local = _EnvGroup(env_fn, self._groups[0])

        spaces = self._call("spaces", only_first=True)
        self.metadata = spaces["metadata"]
        self.render_mode = spaces["render_mode"]
        self.possible_agents = spaces["possible_agents"]
        self.agents = self.possible_agents[:]
        self.observation_spaces = spaces["observation_spaces"]
        self.action_spaces = spaces["action_spaces"]
        self.state_space = spaces["state_space"]
        self._discrete = {
            uid: isinstance(space, gymnasium.spaces.Discrete) for uid, space in self.action_spaces.items()
        }

    def _receive(self, remotes: list[Any]) -> list[Any]:
        """Receive the results of the worker processes, raising the first exception (if any)."""
        results = []
        for remote in remotes:
            try:
                success, result = remote.recv()
            except EOFError:
                success, result = False, RuntimeError("A worker process terminated unexpectedly")
            if not success:
                self.close()
                raise result
            results.append(result)
        return results

    def _call(self, command: str, args_per_group: list[tuple] | None = None, *, only_first: bool = False) -> Any:
        """Call a method of the environment groups.

        :param command: Name of the method.
        :param args_per_group: Arguments for each group. If not given, the method is called without arguments.
        :param only_first: Whether to call the method of the first group only (and return its output as is).
            Otherwise, the outputs (lists) of the groups are concatenated.
        """
        if args_per_group is None:
            args_per_group = [()] * len(self._groups)
        if self._local is not None:
            return getattr(self._local, command)(*args_per_group[0])
        remotes = self._remotes[:1] if only_first else self._remotes
        for remote, args in zip(remotes, args_per_group):
            remote.send((command, args))
        outputs = self._receive(remotes)
        return outputs[0] if only_first else [item for output in outputs for item in output]

    def observation_space(self, agent: str) -> gymnasium.Space:
        """Observation space of the given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gymnasium.Space:
        """Action space of the given agent."""
        return self.action_spaces[agent]

    def _batch(self, values: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        return {uid: np.stack([value[uid] for value in values]) for uid in self.possible_agents}

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict[str, np.ndarray], dict]:
        """Reset all the environment instances.

        :param seed: Seed of the first environment instance (the others are seeded with ``seed`` plus their index).
        :param options: Options passed to the environment instances' reset (also used by the auto-reset).

        :return: Batched observations and info.
        """
        observations, infos = zip(*self._call("reset", [(seed, options)] * len(self._groups)))
        return self._batch(observations), {"envs": list(infos)}

    def step(self, actions: dict[str, np.ndarray]) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, Any],
    ]:
        """Step all the environment instances (and auto-reset those whose agents are all done).

        :param actions: Batched actions (discrete actions are batched with shape ``(num_envs, 1)``).

        :return: Batched observations, rewards, terminations and truncations, and info.
        """
        env_actions = [
            {
                uid: int(actions[uid][i].item()) if self._discrete[uid] else actions[uid][i]
                for uid in self.possible_agents
            }
            for i in range(self.num_envs)
        ]
        outputs = self._call("step", [([env_actions[i] for i in indexes],) for indexes in self._groups])
        observations, rewards, terminations, truncations, infos, final_observations, final_states = zip(*outputs)
        info = {
            "envs": list(infos),
            "final_observations": self._batch(final_observations),
            "final_states": np.stack(final_states),
        }
        return (
            self._batch(observations),
            self._batch(rewards),
            self._batch(terminations),
            self._batch(truncations),
            info,
        )

    def state(self) -> np.ndarray:
        """Batched state of all the environment instances."""
        return np.stack(self._call("state"))

    def render(self) -> Any:
        """Render the first environment instance."""
        return self._call("render", only_first=True)

    def close(self) -> None:
        """Close the environment instances (and stop the worker processes)."""
        if self._closed:
            return
        self._closed = True
        if self._local is not None:
            self._local.close()
            return
        for remote in self._remotes:
            try:
                remote.send(("close", ()))
                remote.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            remote.close()
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
