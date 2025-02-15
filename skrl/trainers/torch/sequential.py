from typing import List, Optional, Union

import copy
import sys
import tqdm

import torch

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch import Trainer


# fmt: off
# [start-config-dict-torch]
SEQUENTIAL_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,            # number of timesteps to train for
    "headless": False,              # whether to use headless mode (no rendering)
    "disable_progressbar": False,   # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,   # whether to close the environment on normal program termination
    "environment_info": "episode",       # key used to get and log environment info
    "stochastic_evaluation": False,      # whether to use actions rather than (deterministic) mean actions during evaluation
}
# [end-config-dict-torch]
# fmt: on


class SequentialTrainer(Trainer):
    def __init__(
        self,
        env: Wrapper,
        agents: Union[Agent, List[Agent]],
        agents_scope: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Sequential trainer for training simultaneous agents one after another.

        .. note::

            Non-simultaneous agents will be trained/evaluated using the base class methods
            :func:`~skrl.trainers.torch.base.Trainer.non_simultaneous_train` and
            :func:`~skrl.trainers.torch.base.Trainer.non_simultaneous_eval` respectively.

        Args:
            env: Environment to train on.
            agents: Agent or sequential agents to train.
            agents_scope: Optional list specifying number of environments for each agent.
                If not provided, environments will be divided equally among sequential agents.
            cfg: Trainer configuration dictionary.
                See :data:`~skrl.trainers.torch.sequential.SEQUENTIAL_TRAINER_DEFAULT_CONFIG` for default values.
        """
        _cfg = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        agents_scope = agents_scope if agents_scope is not None else []
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)

        # init agents
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)

    def train(self) -> None:
        """Train simultaneous agent (single-agent or multi-agent) sequentially.

        .. note::

            Non-simultaneous agents will be trained using the base class method
            :func:`~skrl.trainers.torch.base.Trainer.non_simultaneous_train`.

        This method executes the following steps in loop (:guilabel:`timesteps` times).
        If :guilabel:`disable_progressbar` is false, a progress bar will be shown.

        - Agent's pre-interaction
        - Compute actions
        - Interact with the environment(s)
        - Render scene (if :guilabel:`headless` is false)
        - Record environment transition(s) and agent data
        - Log environment info (if :guilabel:`environment_info` is in ``info``)
        - Agent's post-interaction
        - Reset environment(s)
        """
        # non-simultaneous agents
        if self.num_simultaneous_agents == 1:
            self.non_simultaneous_train()
            return

        # set running mode
        for agent in self.agents:
            agent.set_running_mode("train")

        # reset environments
        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                actions = torch.vstack(
                    [
                        agent.act(
                            observations[scope[0] : scope[1]],
                            states[scope[0] : scope[1]],
                            timestep=timestep,
                            timesteps=self.timesteps,
                        )[0]
                        for agent, scope in zip(self.agents, self.agents_scope)
                    ]
                )

                # step the environments
                next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                next_states = self.env.state()

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                for agent, scope in zip(self.agents, self.agents_scope):
                    agent.record_transition(
                        observations=observations[scope[0] : scope[1]],
                        states=states[scope[0] : scope[1]],
                        actions=actions[scope[0] : scope[1]],
                        rewards=rewards[scope[0] : scope[1]],
                        next_observations=next_observations[scope[0] : scope[1]],
                        next_states=next_states[scope[0] : scope[1]],
                        terminated=terminated[scope[0] : scope[1]],
                        truncated=truncated[scope[0] : scope[1]],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            for agent in self.agents:
                                agent.track_data(f"Info / {k}", v.item())

            # post-interaction
            for agent in self.agents:
                agent.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        observations, infos = self.env.reset()
                        states = self.env.state()
                else:
                    observations = next_observations
                    states = next_states

    def eval(self) -> None:
        """Evaluate simultaneous agent (single-agent or multi-agent) sequentially.

        .. note::

            Non-simultaneous agents will be evaluated using the base class method
            :func:`~skrl.trainers.torch.base.Trainer.non_simultaneous_eval`.

        This method executes the following steps in loop (:guilabel:`timesteps` times).
        If :guilabel:`disable_progressbar` is false, a progress bar will be shown.

        - Agent's pre-interaction
        - Compute actions (stochastic actions if :guilabel:`stochastic_evaluation` is true)
        - Interact with the environment(s)
        - Render scene (if :guilabel:`headless` is false)
        - Record environment transition(s)
        - Log environment info (if :guilabel:`environment_info` is in ``info``)
        - Agent's post-interaction (TensorBoard data writing and checkpoint saving)
        - Reset environment(s)
        """
        # non-simultaneous agents
        if self.num_simultaneous_agents == 1:
            self.non_simultaneous_eval()
            return

        # set running mode
        for agent in self.agents:
            agent.set_running_mode("eval")

        # reset environments
        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                outputs = [
                    agent.act(
                        observations[scope[0] : scope[1]],
                        states[scope[0] : scope[1]],
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )
                    for agent, scope in zip(self.agents, self.agents_scope)
                ]
                actions = torch.vstack(
                    [
                        output[0] if self.stochastic_evaluation else output[-1].get("mean_actions", output[0])
                        for output in outputs
                    ]
                )

                # step the environments
                next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                next_states = self.env.state()
                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                for agent, scope in zip(self.agents, self.agents_scope):
                    agent.record_transition(
                        observations=observations[scope[0] : scope[1]],
                        states=states[scope[0] : scope[1]],
                        actions=actions[scope[0] : scope[1]],
                        rewards=rewards[scope[0] : scope[1]],
                        next_observations=next_observations[scope[0] : scope[1]],
                        next_states=next_states[scope[0] : scope[1]],
                        terminated=terminated[scope[0] : scope[1]],
                        truncated=truncated[scope[0] : scope[1]],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            for agent in self.agents:
                                agent.track_data(f"Info / {k}", v.item())

            # post-interaction (base class, TensorBoard data writing and checkpoint saving)
            for agent in self.agents:
                super(type(agent), agent).post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        observations, infos = self.env.reset()
                        states = self.env.state()
                else:
                    observations = next_observations
                    states = next_states
