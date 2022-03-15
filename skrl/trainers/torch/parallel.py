from typing import Union, List

import torch
import torch.multiprocessing as mp

from ...envs.torch import Wrapper
from ...agents.torch import Agent

from . import Trainer


def fn_processor(process_index, *args):
    print("[INFO] Processor {}: started".format(process_index))

    pipe = args[0][process_index]
    queue = args[1][process_index]
    barrier = args[2]
    scope = args[3][process_index]

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
            agent.init()
            print("[INFO] Processor {}: init agent {} with scope {}".format(process_index, type(agent).__name__, scope))
            barrier.wait()
        
        # execute agent's pre-interaction step
        elif task == "pre_interaction":
            agent.pre_interaction(timestep=msg['timestep'], timesteps=msg['timesteps'])
            barrier.wait()

        # get agent's actions
        elif task == "act":
            _states = queue.get()[scope[0]:scope[1]]
            with torch.no_grad():
                _actions = agent.act(_states, 
                                     inference=True,
                                     timestep=msg['timestep'], 
                                     timesteps=msg['timesteps'])[0]
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
                                        dones=queue.get()[scope[0]:scope[1]],
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
                super(type(agent), agent).record_transition(states=_states, 
                                                            actions=_actions,
                                                            rewards=queue.get()[scope[0]:scope[1]],
                                                            next_states=queue.get()[scope[0]:scope[1]],
                                                            dones=queue.get()[scope[0]:scope[1]],
                                                            timestep=msg['timestep'],
                                                            timesteps=msg['timesteps'])
                super(type(agent), agent).post_interaction(timestep=msg['timestep'], timesteps=msg['timesteps'])
                barrier.wait()


class ParallelTrainer(Trainer):
    def __init__(self, 
                 cfg: dict, 
                 env: Wrapper, 
                 agents: Union[Agent, List[Agent], List[List[Agent]]], 
                 agents_scope : List[int] = []) -> None:
        """Parallel trainer
        
        Train agents in parallel using multiple processes

        :param cfg: Configuration dictionary
        :type cfg: dict
        :param env: Environment to train on
        :type env: skrl.env.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: [])
        :type agents_scope: tuple or list of integers
        """
        super().__init__(cfg, env, agents, agents_scope)

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
        # single agent
        if self.num_agents == 1:
            self.agents.init()
            self.single_agent_train()
            return

        # initialize multiprocessing variables
        queues = []
        producer_pipes = []
        consumer_pipes = []
        barrier = mp.Barrier(self.num_agents + 1)
        processes = []
        
        for i in range(self.num_agents):
            pipe_read, pipe_write = mp.Pipe(duplex=False)
            producer_pipes.append(pipe_write)
            consumer_pipes.append(pipe_read)
            queues.append(mp.Queue())

        # move tensors to shared memory
        for agent in self.agents:
            if agent.memory is not None:
                agent.memory.share_memory()
            for model in agent.models.values():
                model.share_memory()

        # spawn and wait for all processes to start
        for i in range(self.num_agents):
            process = mp.Process(target=fn_processor,
                                 args=(i, consumer_pipes, queues, barrier, self.agents_scope),
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
        states = self.env.reset()
        states.share_memory_()

        for timestep in range(self.initial_timestep, self.timesteps):
            # show progress
            self.show_progress(timestep=timestep, timesteps=self.timesteps)

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
            next_states, rewards, dones, infos = self.env.step(actions)
            
            # render scene
            if not self.headless:
                self.env.render()

            # record the environments' transitions
            with torch.no_grad():
                rewards.share_memory_()
                next_states.share_memory_()
                dones.share_memory_()
                
                for pipe, queue in zip(producer_pipes, queues):
                    pipe.send({"task": "record_transition", "timestep": timestep, "timesteps": self.timesteps})
                    queue.put(rewards)
                    queue.put(next_states)
                    queue.put(dones)
                barrier.wait()

            # post-interaction
            for pipe in producer_pipes:
                pipe.send({"task": "post_interaction", "timestep": timestep, "timesteps": self.timesteps})
            barrier.wait()

            # reset environments
            with torch.no_grad():
                if dones.any():
                    states = self.env.reset()
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
        # single agent
        if self.num_agents == 1:
            self.agents.init()
            self.single_agent_eval()
            return

        # initialize multiprocessing variables
        queues = []
        producer_pipes = []
        consumer_pipes = []
        barrier = mp.Barrier(self.num_agents + 1)
        processes = []
        
        for i in range(self.num_agents):
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
                    model.share_memory()

        # spawn and wait for all processes to start
        for i in range(self.num_agents):
            process = mp.Process(target=fn_processor,
                                 args=(i, consumer_pipes, queues, barrier, self.agents_scope),
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
        states = self.env.reset()
        states.share_memory_()

        for timestep in range(self.initial_timestep, self.timesteps):
            # show progress
            self.show_progress(timestep=timestep, timesteps=self.timesteps)

            # compute actions
            with torch.no_grad():
                for pipe, queue in zip(producer_pipes, queues):
                    pipe.send({"task": "act", "timestep": timestep, "timesteps": self.timesteps})
                    queue.put(states)

                barrier.wait()
                actions = torch.vstack([queue.get() for queue in queues])
                
            # step the environments
            next_states, rewards, dones, infos = self.env.step(actions)
            
            # render scene
            if not self.headless:
                self.env.render()

            with torch.no_grad():
                # write data to TensorBoard
                rewards.share_memory_()
                next_states.share_memory_()
                dones.share_memory_()
                
                for pipe, queue in zip(producer_pipes, queues):
                    pipe.send({"task": "eval-record_transition-post_interaction", 
                               "timestep": timestep, 
                               "timesteps": self.timesteps})
                    queue.put(rewards)
                    queue.put(next_states)
                    queue.put(dones)
                barrier.wait()

                # reset environments
                if dones.any():
                    states = self.env.reset()
                    states.share_memory_()
                else:
                    states.copy_(next_states)

        # terminate processes
        for pipe in producer_pipes:
            pipe.send({"task": "terminate"})
        
        # join processes
        for process in processes:
            process.join()
        