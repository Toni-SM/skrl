from typing import Union, List

from skrl.envs.torch import Wrapper   # from ...envs.torch import Wrapper
from skrl.agents.torch import Agent   # from ...agents.torch import Agent

from skrl.trainers.torch import Trainer       # from . import Trainer


class CustomTrainer(Trainer):
    def __init__(self, cfg: dict, env: Wrapper, agents: Union[Agent, List[Agent], List[List[Agent]]], agents_scope : List[int] = []) -> None:
        """
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

    def start(self) -> None:
        """Start training
        """
        # ================================
        # - run training loop
        #   + call agents.pre_interaction(...)
        #   + sample actions using agents.act(...)
        #   + step environment using env.step(...)
        #   + record environment transition in memory using agents.record_transition(...)
        #   + call agents.post_interaction(...)
        # ================================