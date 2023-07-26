from typing import List, Optional, Union

import copy
import tqdm

import torch
import torch.multiprocessing as mp

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch import Trainer


PARALLEL_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,            # number of timesteps to train for
    "headless": False,              # whether to use headless mode (no rendering)
    "disable_progressbar": False,   # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,   # whether to close the environment on normal program termination
}


def fn_processor(process_index, *args):
    print(f"[INFO] Processor {process_index}: started")

    pipe = args[0][process_index]
    queue = args[1][process_index]
    barrier = args[2]
    scope = args[3][process_index]
    trainer_cfg = args[4]

    agent = None
    _states = None
    _actions = None

    # wait for the main process to start all the workers
    barrier.wait()

    while True:
        msg = pipe.recv()
        task = msg['task']

        # terminate process
        if task == 'terminate':
            break

        # initialize agent
        elif task == 'init':
            agent = queue.get()
            agent.init(trainer_cfg=trainer_cfg)
            print(f"[INFO] Processor {process_index}: init agent {type(agent).__name__} with scope {scope}")
            barrier.wait()

        # execute agent's pre-interaction step
        elif task == "pre_interaction":
            agent.pre_interaction(timestep=msg['timestep'], timesteps=msg['timesteps'])
            barrier.wait()

        # get agent's actions
        elif task == "act":
            _states = queue.get()[scope[0]:scope[1]]
            with torch.no_grad():
                _actions = agent.act(_states, timestep=msg['timestep'], timesteps=msg['timesteps'])[0]
                if not _actions.is_cuda:
                    _actions.share_memory_()
                queue.put(_actions)
                barrier.wait()

        # record agent's experience
        elif task == "record_transition":
            with torch.no_grad():
                agent.record_transition(states=_states,
                                        actions=_actions,
                                        rewards=queue.get()[scope[0]:scope[1]],
                                        next_states=queue.get()[scope[0]:scope[1]],
                                        terminated=queue.get()[scope[0]:scope[1]],
                                        truncated=queue.get()[scope[0]:scope[1]],
                                        infos=queue.get(),
                                        timestep=msg['timestep'],
                                        timesteps=msg['timesteps'])
                barrier.wait()

        # execute agent's post-interaction step
        elif task == "post_interaction":
            agent.post_interaction(timestep=msg['timestep'], timesteps=msg['timesteps'])
            barrier.wait()

        # write data to TensorBoard (evaluation)
        elif task == "eval-record_transition-post_interaction":
            with torch.no_grad():
                agent.record_transition(states=_states,
                                        actions=_actions,
                                        rewards=queue.get()[scope[0]:scope[1]],
                                        next_states=queue.get()[scope[0]:scope[1]],
                                        terminated=queue.get()[scope[0]:scope[1]],
                                        truncated=queue.get()[scope[0]:scope[1]],
                                        infos=queue.get(),
                                        timestep=msg['timestep'],
                                        timesteps=msg['timesteps'])
                super(type(agent), agent).post_interaction(timestep=msg['timestep'], timesteps=msg['timesteps'])
                barrier.wait()


class ParallelTrainer(Trainer):
    def __init__(self,
                 env: Wrapper,
                 agents: Union[Agent, List[Agent]],
                 agents_scope: Optional[List[int]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Parallel trainer

        Train agents in parallel using multiple processes

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``).
                    See PARALLEL_TRAINER_DEFAULT_CONFIG for default values
        :type cfg: dict, optional
        """
        _cfg = copy.deepcopy(PARALLEL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        agents_scope = agents_scope if agents_scope is not None else []
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)

        mp.set_start_method(method='spawn', force=True)

    def train(self) -> None:
        """Train the agents in parallel

        This method executes the following steps in loop:

        - Pre-interaction (parallel)
        - Compute actions (in parallel)
        - Interact with the environments
        - Render scene
        - Record transitions (in parallel)
        - Post-interaction (in parallel)
        - Reset environments
        """
        # set running mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("train")
        else:
            self.agents.set_running_mode("train")

        # non-simultaneous agents
        if self.num_simultaneous_agents == 1:
            self.agents.init(trainer_cfg=self.cfg)
            # single-agent
            if self.env.num_agents == 1:
                self.single_agent_train()
            # multi-agent
            else:
                self.multi_agent_train()
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
            process = mp.Process(target=fn_processor,
                                 args=(i, consumer_pipes, queues, barrier, self.agents_scope, self.cfg),
                                 daemon=True)
            processes.append(process)
            process.start()
        barrier.wait()

        # initialize agents
        for pipe, queue, agent in zip(producer_pipes, queues, self.agents):
            pipe.send({'task': 'init'})
            queue.put(agent)
        barrier.wait()

        # reset env
        states, infos = self.env.reset()
        if not states.is_cuda:
            states.share_memory_()

        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar):

            # pre-interaction
            for pipe in producer_pipes:
                pipe.send({"task": "pre_interaction", "timestep": timestep, "timesteps": self.timesteps})
            barrier.wait()

            # compute actions
            with torch.no_grad():
                for pipe, queue in zip(producer_pipes, queues):
                    pipe.send({"task": "act", "timestep": timestep, "timesteps": self.timesteps})
                    queue.put(states)

                barrier.wait()
                actions = torch.vstack([queue.get() for queue in queues])

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                if not rewards.is_cuda:
                    rewards.share_memory_()
                if not next_states.is_cuda:
                    next_states.share_memory_()
                if not terminated.is_cuda:
                    terminated.share_memory_()
                if not truncated.is_cuda:
                    truncated.share_memory_()

                for pipe, queue in zip(producer_pipes, queues):
                    pipe.send({"task": "record_transition", "timestep": timestep, "timesteps": self.timesteps})
                    queue.put(rewards)
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
            with torch.no_grad():
                if terminated.any() or truncated.any():
                    states, infos = self.env.reset()
                    if not states.is_cuda:
                        states.share_memory_()
                else:
                    states.copy_(next_states)

        # terminate processes
        for pipe in producer_pipes:
            pipe.send({"task": "terminate"})

        # join processes
        for process in processes:
            process.join()

    def eval(self) -> None:
        """Evaluate the agents sequentially

        This method executes the following steps in loop:

        - Compute actions (in parallel)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        # set running mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("eval")
        else:
            self.agents.set_running_mode("eval")

        # non-simultaneous agents
        if self.num_simultaneous_agents == 1:
            self.agents.init(trainer_cfg=self.cfg)
            # single-agent
            if self.env.num_agents == 1:
                self.single_agent_eval()
            # multi-agent
            else:
                self.multi_agent_eval()
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
            process = mp.Process(target=fn_processor,
                                 args=(i, consumer_pipes, queues, barrier, self.agents_scope, self.cfg),
                                 daemon=True)
            processes.append(process)
            process.start()
        barrier.wait()

        # initialize agents
        for pipe, queue, agent in zip(producer_pipes, queues, self.agents):
            pipe.send({'task': 'init'})
            queue.put(agent)
        barrier.wait()

        # reset env
        states, infos = self.env.reset()
        if not states.is_cuda:
            states.share_memory_()

        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar):

            # compute actions
            with torch.no_grad():
                for pipe, queue in zip(producer_pipes, queues):
                    pipe.send({"task": "act", "timestep": timestep, "timesteps": self.timesteps})
                    queue.put(states)

                barrier.wait()
                actions = torch.vstack([queue.get() for queue in queues])

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                if not rewards.is_cuda:
                    rewards.share_memory_()
                if not next_states.is_cuda:
                    next_states.share_memory_()
                if not terminated.is_cuda:
                    terminated.share_memory_()
                if not truncated.is_cuda:
                    truncated.share_memory_()

                for pipe, queue in zip(producer_pipes, queues):
                    pipe.send({"task": "eval-record_transition-post_interaction",
                               "timestep": timestep,
                               "timesteps": self.timesteps})
                    queue.put(rewards)
                    queue.put(next_states)
                    queue.put(terminated)
                    queue.put(truncated)
                    queue.put(infos)
                barrier.wait()

                # reset environments
                if terminated.any() or truncated.any():
                    states, infos = self.env.reset()
                    if not states.is_cuda:
                        states.share_memory_()
                else:
                    states.copy_(next_states)

        # terminate processes
        for pipe in producer_pipes:
            pipe.send({"task": "terminate"})

        # join processes
        for process in processes:
            process.join()
