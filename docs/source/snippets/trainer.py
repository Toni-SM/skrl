# [start-base]
from typing import Union, List

from skrl.envs.torch import Wrapper   # from ...envs.torch import Wrapper
from skrl.agents.torch import Agent   # from ...agents.torch import Agent

from skrl.trainers.torch import Trainer       # from . import Trainer


class CustomTrainer(Trainer):
    def __init__(self, 
                 cfg: dict, 
                 env: Wrapper, 
                 agents: Union[Agent, List[Agent], List[List[Agent]]], 
                 agents_scope : List[int] = []) -> None:
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

        # ================================
        # - init agents
        # ================================

    def train(self) -> None:
        """Train the agents
        """
        # ================================
        # - run training loop
        #   + call agents.pre_interaction(...)
        #   + compute actions using agents.act(...)
        #   + step environment using env.step(...)
        #   + render scene using env.render(...)
        #   + record environment transition in memory using agents.record_transition(...)
        #   + call agents.post_interaction(...)
        #   + reset environment using env.reset(...)
        # ================================

    def eval(self) -> None:
        """Evaluate the agents
        """
        # ================================
        # - run evaluation loop
        #   + compute actions using agents.act(...)
        #   + step environment using env.step(...)
        #   + render scene using env.render(...)
        #   + call agents.post_interaction(...) parent method to write data to TensorBoard
        #   + reset environment using env.reset(...)
        # ================================
# [end-base]

# =============================================================================

# [start-sequential]
from skrl.trainers.torch import SequentialTrainer

# asuming there is an environment called 'env'
# and an agent or a list of agents called 'agents'

# create a sequential trainer
cfg = {"timesteps": 50000, "headless": False}
trainer = SequentialTrainer(cfg=cfg, env=env, agents=agents)

# train the agent(s)
trainer.train()

# evaluate the agent(s)
trainer.eval()
# [end-sequential]

# =============================================================================

# [start-parallel]
from skrl.trainers.torch import ParallelTrainer

# asuming there is an environment called 'env'
# and an agent or a list of agents called 'agents'

# create a sequential trainer
cfg = {"timesteps": 50000, "headless": False}
trainer = ParallelTrainer(cfg=cfg, env=env, agents=agents)

# train the agent(s)
trainer.train()

# evaluate the agent(s)
trainer.eval()
# [end-parallel]
