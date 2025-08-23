from typing import List, Optional, Union

import copy
import sys
import tqdm

import torch
import torch.multiprocessing as mp

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.multi_agents.torch import MultiAgent
from skrl.trainers.torch import Trainer
from skrl.utils import ScopedTimer


# fmt: off
# [start-config-dict-torch]
PARALLEL_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,            # number of timesteps to train for
    "headless": False,              # whether to use headless mode (no rendering)
    "disable_progressbar": False,   # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,   # whether to close the environment on normal program termination
    "environment_info": "episode",       # key used to get and log environment info
    "stochastic_evaluation": False,      # whether to use actions rather than (deterministic) mean actions during evaluation
}
# [end-config-dict-torch]
# fmt: on


def fn_processor(process_index, *args):
    print(f"[INFO] Processor {process_index}: started")

    pipe = args[0][process_index]
    queue = args[1][process_index]
    barrier = args[2]
    scope = args[3][process_index]
    trainer_cfg = args[4]

    agent = None
    _observations = None
    _states = None
    _actions = None

    # wait for the main process to start all the workers
    barrier.wait()

    while True:
        msg = pipe.recv()
        task = msg["task"]

        # terminate process
        if task == "terminate":
            break

        # initialize agent
        elif task == "init":
            agent = queue.get()
            agent.init(trainer_cfg=trainer_cfg)
            print(f"[INFO] Processor {process_index}: init agent {type(agent).__name__} with scope {scope}")
            barrier.wait()

        # execute agent's pre-interaction step
        elif task == "pre_interaction":
            agent.pre_interaction(timestep=msg["timestep"], timesteps=msg["timesteps"])
            barrier.wait()

        # get agent's actions
        elif task == "act":
            _observations = queue.get()[scope[0] : scope[1]]
            _states = queue.get()
            if _states is not None:
                _states = _states[scope[0] : scope[1]]
            with torch.no_grad():
                with ScopedTimer() as timer:
                    _actions, _outputs = agent.act(
                        _observations, _states, timestep=msg["timestep"], timesteps=msg["timesteps"]
                    )
                    agent.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)
                if not msg.get("stochastic_evaluation", True):
                    _actions = _outputs.get("mean_actions", _actions)
                if not _actions.is_cuda:
                    _actions.share_memory_()
                queue.put(_actions)
                barrier.wait()

        # record agent's experience
        elif task == "record_transition":
            agent.track_data("Stats / Env stepping time (ms)", msg["env_stepping_time_ms"])
            with torch.no_grad():
                rewards = queue.get()[scope[0] : scope[1]]
                next_observations = queue.get()[scope[0] : scope[1]]
                next_states = queue.get()
                if next_states is not None:
                    next_states = next_states[scope[0] : scope[1]]
                terminated = queue.get()[scope[0] : scope[1]]
                truncated = queue.get()[scope[0] : scope[1]]
                infos = queue.get()
                agent.record_transition(
                    observations=_observations,
                    states=_states,
                    actions=_actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=msg["timestep"],
                    timesteps=msg["timesteps"],
                )
                barrier.wait()

        # execute agent's post-interaction step
        elif task == "post_interaction":
            agent.post_interaction(timestep=msg["timestep"], timesteps=msg["timesteps"])
            barrier.wait()

        # write data to TensorBoard (evaluation)
        elif task == "eval-record_transition-post_interaction":
            agent.track_data("Stats / Env stepping time (ms)", msg["env_stepping_time_ms"])
            with torch.no_grad():
                rewards = queue.get()[scope[0] : scope[1]]
                next_observations = queue.get()[scope[0] : scope[1]]
                next_states = queue.get()
                if next_states is not None:
                    next_states = next_states[scope[0] : scope[1]]
                terminated = queue.get()[scope[0] : scope[1]]
                truncated = queue.get()[scope[0] : scope[1]]
                infos = queue.get()
                agent.record_transition(
                    observations=_observations,
                    states=_states,
                    actions=_actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=msg["timestep"],
                    timesteps=msg["timesteps"],
                )
                super(agent.__class__, agent).post_interaction(timestep=msg["timestep"], timesteps=msg["timesteps"])
                barrier.wait()


class ParallelTrainer(Trainer):
    def __init__(
        self,
        *,
        env: Union[Wrapper, MultiAgentEnvWrapper],
        agents: Union[Agent, MultiAgent, List[Agent], List[MultiAgent]],
        scopes: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Parallel trainer.

        Train agents in parallel using multiple processes.

        :param env: Environment to train/evaluate on.
        :param agents: Agent(s) to train/evaluate.
        :param scopes: Number of environments for each simultaneous agent to train/evaluate on.
        :param cfg: Configuration dictionary.
        """
        _cfg = copy.deepcopy(PARALLEL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        scopes = scopes if scopes is not None else []
        super().__init__(env=env, agents=agents, scopes=scopes, cfg=_cfg)

        mp.set_start_method(method="spawn", force=True)

    def train(self) -> None:
        """Train agents in parallel.

        This method executes the following steps in loop:

        - Pre-interaction (parallel)
        - Compute actions (in parallel)
        - Interact with the environments
        - Render environments
        - Record transitions (in parallel)
        - Post-interaction (in parallel)
        - Reset environments
        """
        # set mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.enable_training_mode(True)
        else:
            self.agents.enable_training_mode(True)

        # non-simultaneous agents
        if self.num_simultaneous_agents == 1:
            self.agents.init(trainer_cfg=self.cfg)
            super().train()
            return

        # initialize multiprocessing variables
        queues = []
        producer_pipes = []
        consumer_pipes = []
        barrier = mp.Barrier(self.num_simultaneous_agents + 1)
        processes = []

        for i in range(self.num_simultaneous_agents):
            pipe_read, pipe_write = mp.Pipe(duplex=False)
            producer_pipes.append(pipe_write)
            consumer_pipes.append(pipe_read)
            queues.append(mp.Queue())

        # move tensors to shared memory
        for agent in self.agents:
            if agent.memory is not None:
                agent.memory.share_memory()
            for model in agent.models.values():
                try:
                    model.share_memory()
                except RuntimeError:
                    pass

        # spawn and wait for all processes to start
        for i in range(self.num_simultaneous_agents):
            process = mp.Process(
                target=fn_processor, args=(i, consumer_pipes, queues, barrier, self.scopes, self.cfg), daemon=True
            )
            processes.append(process)
            process.start()
        barrier.wait()

        # initialize agents
        for pipe, queue, agent in zip(producer_pipes, queues, self.agents):
            pipe.send({"task": "init"})
            queue.put(agent)
        barrier.wait()

        # reset the environments
        observations, infos = self.env.reset()
        states = self.env.state()
        if not observations.is_cuda:
            observations.share_memory_()
        if states is not None and not states.is_cuda:
            states.share_memory_()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            for pipe in producer_pipes:
                pipe.send({"task": "pre_interaction", "timestep": timestep, "timesteps": self.timesteps})
            barrier.wait()

            # compute actions
            with torch.no_grad():
                for pipe, queue in zip(producer_pipes, queues):
                    pipe.send(
                        {
                            "task": "act",
                            "timestep": timestep,
                            "timesteps": self.timesteps,
                        }
                    )
                    queue.put(observations)
                    queue.put(states)

                barrier.wait()
                actions = torch.vstack([queue.get() for queue in queues])

                # step the environments
                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                    next_states = self.env.state()

                # render the environments
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                if not rewards.is_cuda:
                    rewards.share_memory_()
                if not next_observations.is_cuda:
                    next_observations.share_memory_()
                if next_states is not None and not next_states.is_cuda:
                    next_states.share_memory_()
                if not terminated.is_cuda:
                    terminated.share_memory_()
                if not truncated.is_cuda:
                    truncated.share_memory_()

                for pipe, queue in zip(producer_pipes, queues):
                    pipe.send(
                        {
                            "task": "record_transition",
                            "timestep": timestep,
                            "timesteps": self.timesteps,
                            "env_stepping_time_ms": timer.elapsed_time_ms,
                        }
                    )
                    queue.put(rewards)
                    queue.put(next_observations)
                    queue.put(next_states)
                    queue.put(terminated)
                    queue.put(truncated)
                    queue.put(infos)
                barrier.wait()

            # post-interaction
            for pipe in producer_pipes:
                pipe.send({"task": "post_interaction", "timestep": timestep, "timesteps": self.timesteps})
            barrier.wait()

            # reset environments
            # - parallel/vectorized environments (single or multi-agent)
            if self.env.num_envs > 1:
                observations.copy_(next_observations)
                if states is not None:
                    states.copy_(next_states)
            # - single environment
            else:
                raise RuntimeError("Parallel trainer is not supported for single environment")

        # terminate processes
        for pipe in producer_pipes:
            pipe.send({"task": "terminate"})

        # join processes
        for process in processes:
            process.join()

    def eval(self) -> None:
        """Evaluate agents sequentially.

        This method executes the following steps in loop:

        - Pre-interaction (in parallel)
        - Compute actions (in parallel)
        - Interact with the environments
        - Render environments
        - Record transitions (in parallel)
        - Reset environments
        """
        # set running mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.enable_training_mode(False)
        else:
            self.agents.enable_training_mode(False)

        # non-simultaneous agents
        if self.num_simultaneous_agents == 1:
            self.agents.init(trainer_cfg=self.cfg)
            super().eval()
            return

        # initialize multiprocessing variables
        queues = []
        producer_pipes = []
        consumer_pipes = []
        barrier = mp.Barrier(self.num_simultaneous_agents + 1)
        processes = []

        for i in range(self.num_simultaneous_agents):
            pipe_read, pipe_write = mp.Pipe(duplex=False)
            producer_pipes.append(pipe_write)
            consumer_pipes.append(pipe_read)
            queues.append(mp.Queue())

        # move tensors to shared memory
        for agent in self.agents:
            if agent.memory is not None:
                agent.memory.share_memory()
            for model in agent.models.values():
                if model is not None:
                    try:
                        model.share_memory()
                    except RuntimeError:
                        pass

        # spawn and wait for all processes to start
        for i in range(self.num_simultaneous_agents):
            process = mp.Process(
                target=fn_processor, args=(i, consumer_pipes, queues, barrier, self.scopes, self.cfg), daemon=True
            )
            processes.append(process)
            process.start()
        barrier.wait()

        # initialize agents
        for pipe, queue, agent in zip(producer_pipes, queues, self.agents):
            pipe.send({"task": "init"})
            queue.put(agent)
        barrier.wait()

        # reset the environments
        observations, infos = self.env.reset()
        states = self.env.state()
        if not observations.is_cuda:
            observations.share_memory_()
        if states is not None and not states.is_cuda:
            states.share_memory_()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            for pipe in producer_pipes:
                pipe.send({"task": "pre_interaction", "timestep": timestep, "timesteps": self.timesteps})
            barrier.wait()

            # compute actions
            with torch.no_grad():
                for pipe, queue in zip(producer_pipes, queues):
                    pipe.send(
                        {
                            "task": "act",
                            "timestep": timestep,
                            "timesteps": self.timesteps,
                            "stochastic_evaluation": self.stochastic_evaluation,
                        }
                    )
                    queue.put(observations)
                    queue.put(states)

                barrier.wait()
                actions = torch.vstack([queue.get() for queue in queues])

                # step the environments
                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                    next_states = self.env.state()

                # render the environments
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                if not rewards.is_cuda:
                    rewards.share_memory_()
                if not next_observations.is_cuda:
                    next_observations.share_memory_()
                if next_states is not None and not next_states.is_cuda:
                    next_states.share_memory_()
                if not terminated.is_cuda:
                    terminated.share_memory_()
                if not truncated.is_cuda:
                    truncated.share_memory_()

            # post-interaction
            for pipe, queue in zip(producer_pipes, queues):
                pipe.send(
                    {
                        "task": "eval-record_transition-post_interaction",
                        "timestep": timestep,
                        "timesteps": self.timesteps,
                        "env_stepping_time_ms": timer.elapsed_time_ms,
                    }
                )
                queue.put(rewards)
                queue.put(next_observations)
                queue.put(next_states)
                queue.put(terminated)
                queue.put(truncated)
                queue.put(infos)
            barrier.wait()

            # reset environments
            # - parallel/vectorized environments (single or multi-agent)
            if self.env.num_envs > 1:
                observations.copy_(next_observations)
                if states is not None:
                    states.copy_(next_states)
            # - single environment
            else:
                raise RuntimeError("Parallel trainer is not supported for single environment")

        # terminate processes
        for pipe in producer_pipes:
            pipe.send({"task": "terminate"})

        # join processes
        for process in processes:
            process.join()
