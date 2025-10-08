from typing import List, Optional, Union

import contextlib
import copy
import dataclasses
import sys
import tqdm

from skrl.agents.warp import Agent
from skrl.envs.wrappers.warp import Wrapper
from skrl.trainers.warp import Trainer, TrainerCfg
from skrl.utils import ScopedTimer


@dataclasses.dataclass(kw_only=True)
class SequentialTrainerCfg(TrainerCfg):
    """Configuration for the sequential trainer."""


class SequentialTrainer(Trainer):
    def __init__(
        self,
        *,
        env: Wrapper,
        agents: Union[Agent, List[Agent]],
        scopes: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Sequential trainer.

        Train agents sequentially, i.e., one after the other, in each interaction with the environment.

        :param env: Environment to train/evaluate on.
        :param agents: Agent(s) to train/evaluate.
        :param scopes: Number of environments for each simultaneous agent to train/evaluate on.
        :param cfg: Configuration dictionary.
        """
        self.cfg: SequentialTrainerCfg
        super().__init__(
            env=env,
            agents=agents,
            scopes=scopes if scopes is not None else [],
            cfg=SequentialTrainerCfg(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # init agents
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)

    def train(self) -> None:
        """Train agents sequentially.

        This method executes the following steps in loop:

        - Pre-interaction (sequentially)
        - Compute actions (sequentially)
        - Interact with the environments
        - Render environments
        - Record transitions (sequentially)
        - Post-interaction (sequentially)
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
            super().train()
            return

        raise NotImplementedError

    def eval(self) -> None:
        """Evaluate agents sequentially.

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions (sequentially)
        - Interact with the environments
        - Render environments
        - Record transitions
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
            super().eval()
            return

        raise NotImplementedError
