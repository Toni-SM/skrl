from typing import Any, List, Optional, Tuple, Union

import contextlib
import copy
import sys
import tqdm

import jax
import jax.numpy as jnp
import numpy as np

from skrl.agents.jax import Agent
from skrl.envs.wrappers.jax import Wrapper
from skrl.trainers.jax import Trainer


# fmt: off
# [start-config-dict-jax]
STEP_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,            # number of timesteps to train for
    "headless": False,              # whether to use headless mode (no rendering)
    "disable_progressbar": False,   # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,   # whether to close the environment on normal program termination
    "environment_info": "episode",       # key used to get and log environment info
    "stochastic_evaluation": False,      # whether to use actions rather than (deterministic) mean actions during evaluation
}
# [end-config-dict-jax]
# fmt: on


class StepTrainer(Trainer):
    def __init__(
        self,
        env: Wrapper,
        agents: Union[Agent, List[Agent]],
        agents_scope: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Step-by-step trainer

        Train agents by controlling the training/evaluation loop step by step

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.jax.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``).
                    See STEP_TRAINER_DEFAULT_CONFIG for default values
        :type cfg: dict, optional
        """
        _cfg = copy.deepcopy(STEP_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        agents_scope = agents_scope if agents_scope is not None else []
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)

        # init agents
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)

        self._timestep = 0
        self._progress = None

        self.states = None

    def train(self, timestep: Optional[int] = None, timesteps: Optional[int] = None) -> Tuple[
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Any,
    ]:
        """Execute a training iteration

        This method executes the following steps once:

        - Pre-interaction (sequentially if num_simultaneous_agents > 1)
        - Compute actions (sequentially if num_simultaneous_agents > 1)
        - Interact with the environments
        - Render scene
        - Record transitions (sequentially if num_simultaneous_agents > 1)
        - Post-interaction (sequentially if num_simultaneous_agents > 1)
        - Reset environments

        :param timestep: Current timestep (default: ``None``).
                         If None, the current timestep will be carried by an internal variable
        :type timestep: int, optional
        :param timesteps: Total number of timesteps (default: ``None``).
                          If None, the total number of timesteps is obtained from the trainer's config
        :type timesteps: int, optional

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of np.ndarray or jax.Array and any other info
        """
        if timestep is None:
            self._timestep += 1
            timestep = self._timestep
        timesteps = self.timesteps if timesteps is None else timesteps

        if self._progress is None:
            self._progress = tqdm.tqdm(total=timesteps, disable=self.disable_progressbar, file=sys.stdout)
        self._progress.update(n=1)

        # hack to simplify code
        if self.num_simultaneous_agents == 1:
            self.agents = [self.agents]

        # set running mode
        for agent in self.agents:
            agent.set_running_mode("train")

        # reset env
        if self.states is None:
            self.states, infos = self.env.reset()

        # pre-interaction
        for agent in self.agents:
            agent.pre_interaction(timestep=timestep, timesteps=timesteps)

        with contextlib.nullcontext():
            # compute actions
            actions = jnp.vstack(
                [
                    agent.act(self.states[scope[0] : scope[1]], timestep=timestep, timesteps=timesteps)[0]
                    for agent, scope in zip(self.agents, self.agents_scope)
                ]
            )

            # step the environments
            next_states, rewards, terminated, truncated, infos = self.env.step(actions)

            # render scene
            if not self.headless:
                self.env.render()

            # record the environments' transitions
            for agent, scope in zip(self.agents, self.agents_scope):
                agent.record_transition(
                    states=self.states[scope[0] : scope[1]],
                    actions=actions[scope[0] : scope[1]],
                    rewards=rewards[scope[0] : scope[1]],
                    next_states=next_states[scope[0] : scope[1]],
                    terminated=terminated[scope[0] : scope[1]],
                    truncated=truncated[scope[0] : scope[1]],
                    infos=infos,
                    timestep=timestep,
                    timesteps=timesteps,
                )

        # post-interaction
        for agent in self.agents:
            agent.post_interaction(timestep=timestep, timesteps=timesteps)

        # reset environments
        if terminated.any() or truncated.any():
            with contextlib.nullcontext():
                self.states, infos = self.env.reset()
        else:
            self.states = next_states

        return next_states, rewards, terminated, truncated, infos

    def eval(self, timestep: Optional[int] = None, timesteps: Optional[int] = None) -> Tuple[
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Any,
    ]:
        """Evaluate the agents sequentially

        This method executes the following steps in loop:

        - Compute actions (sequentially if num_simultaneous_agents > 1)
        - Interact with the environments
        - Render scene
        - Reset environments

        :param timestep: Current timestep (default: ``None``).
                         If None, the current timestep will be carried by an internal variable
        :type timestep: int, optional
        :param timesteps: Total number of timesteps (default: ``None``).
                          If None, the total number of timesteps is obtained from the trainer's config
        :type timesteps: int, optional

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of np.ndarray or jax.Array and any other info
        """
        if timestep is None:
            self._timestep += 1
            timestep = self._timestep
        timesteps = self.timesteps if timesteps is None else timesteps

        if self._progress is None:
            self._progress = tqdm.tqdm(total=timesteps, disable=self.disable_progressbar, file=sys.stdout)
        self._progress.update(n=1)

        # hack to simplify code
        if self.num_simultaneous_agents == 1:
            self.agents = [self.agents]

        # set running mode
        for agent in self.agents:
            agent.set_running_mode("eval")

        # reset env
        if self.states is None:
            self.states, infos = self.env.reset()

        # pre-interaction
        for agent in self.agents:
            agent.pre_interaction(timestep=timestep, timesteps=timesteps)

        with contextlib.nullcontext():
            # compute actions
            outputs = [
                agent.act(self.states[scope[0] : scope[1]], timestep=timestep, timesteps=timesteps)
                for agent, scope in zip(self.agents, self.agents_scope)
            ]
            actions = jnp.vstack(
                [
                    output[0] if self.stochastic_evaluation else output[-1].get("mean_actions", output[0])
                    for output in outputs
                ]
            )

            # step the environments
            next_states, rewards, terminated, truncated, infos = self.env.step(actions)

            # render scene
            if not self.headless:
                self.env.render()

            # write data to TensorBoard
            for agent, scope in zip(self.agents, self.agents_scope):
                agent.record_transition(
                    states=self.states[scope[0] : scope[1]],
                    actions=actions[scope[0] : scope[1]],
                    rewards=rewards[scope[0] : scope[1]],
                    next_states=next_states[scope[0] : scope[1]],
                    terminated=terminated[scope[0] : scope[1]],
                    truncated=truncated[scope[0] : scope[1]],
                    infos=infos,
                    timestep=timestep,
                    timesteps=timesteps,
                )

        # post-interaction
        for agent in self.agents:
            super(type(agent), agent).post_interaction(timestep=timestep, timesteps=timesteps)

        # reset environments
        if terminated.any() or truncated.any():
            with contextlib.nullcontext():
                self.states, infos = self.env.reset()
        else:
            self.states = next_states

        return next_states, rewards, terminated, truncated, infos
