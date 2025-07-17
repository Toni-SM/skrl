from typing import List, Optional, Union

import contextlib
import copy
import sys
import tqdm

from skrl.agents.warp import Agent
from skrl.envs.wrappers.warp import Wrapper
from skrl.trainers.warp import Trainer


# fmt: off
# [start-config-dict-warp]
SEQUENTIAL_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,            # number of timesteps to train for
    "headless": False,              # whether to use headless mode (no rendering)
    "disable_progressbar": False,   # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,   # whether to close the environment on normal program termination
    "environment_info": "episode",       # key used to get and log environment info
    "stochastic_evaluation": False,      # whether to use actions rather than (deterministic) mean actions during evaluation
}
# [end-config-dict-warp]
# fmt: on


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
        _cfg = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        scopes = scopes if scopes is not None else []
        super().__init__(env=env, agents=agents, scopes=scopes, cfg=_cfg)

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
