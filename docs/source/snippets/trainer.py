# [pytorch-start-base]
from typing import Union, List, Optional

import copy

from skrl.envs.wrappers.torch import Wrapper
from skrl.agents.torch import Agent

from skrl.trainers.torch import Trainer


CUSTOM_DEFAULT_CONFIG = {
    "timesteps": 100000,            # number of timesteps to train for
    "headless": False,              # whether to use headless mode (no rendering)
    "disable_progressbar": False,   # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,   # whether to close the environment on normal program termination
}


class CustomTrainer(Trainer):
    def __init__(self,
                 env: Wrapper,
                 agents: Union[Agent, List[Agent], List[List[Agent]]],
                 agents_scope: Optional[List[int]] = None,
                 cfg: Optional[dict] = None) -> None:
        """
        :param env: Environment to train on
        :type env: skrl.envs.wrappers.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: [])
        :type agents_scope: tuple or list of integers
        :param cfg: Configuration dictionary
        :type cfg: dict, optional
        """
        _cfg = copy.deepcopy(CUSTOM_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        agents_scope = agents_scope if agents_scope is not None else []
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)

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
# [pytorch-end-base]


# [jax-start-base]
from typing import Union, List, Optional

import copy

from skrl.envs.wrappers.jax import Wrapper
from skrl.agents.jax import Agent

from skrl.trainers.jax import Trainer


CUSTOM_DEFAULT_CONFIG = {
    "timesteps": 100000,            # number of timesteps to train for
    "headless": False,              # whether to use headless mode (no rendering)
    "disable_progressbar": False,   # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,   # whether to close the environment on normal program termination
}


class CustomTrainer(Trainer):
    def __init__(self,
                 env: Wrapper,
                 agents: Union[Agent, List[Agent], List[List[Agent]]],
                 agents_scope: Optional[List[int]] = None,
                 cfg: Optional[dict] = None) -> None:
        """
        :param env: Environment to train on
        :type env: skrl.envs.wrappers.jax.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: [])
        :type agents_scope: tuple or list of integers
        :param cfg: Configuration dictionary
        :type cfg: dict, optional
        """
        _cfg = copy.deepcopy(CUSTOM_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        agents_scope = agents_scope if agents_scope is not None else []
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)

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
# [jax-end-base]

# =============================================================================

# [pytorch-start-sequential]
from skrl.trainers.torch import SequentialTrainer

# assuming there is an environment called 'env'
# and an agent or a list of agents called 'agents'

# create a sequential trainer
cfg = {"timesteps": 50000, "headless": False}
trainer = SequentialTrainer(env=env, agents=agents, cfg=cfg)

# train the agent(s)
trainer.train()

# evaluate the agent(s)
trainer.eval()
# [pytorch-end-sequential]


# [jax-start-sequential]
from skrl.trainers.jax import SequentialTrainer

# assuming there is an environment called 'env'
# and an agent or a list of agents called 'agents'

# create a sequential trainer
cfg = {"timesteps": 50000, "headless": False}
trainer = SequentialTrainer(env=env, agents=agents, cfg=cfg)

# train the agent(s)
trainer.train()

# evaluate the agent(s)
trainer.eval()
# [jax-end-sequential]

# =============================================================================

# [pytorch-start-parallel]
from skrl.trainers.torch import ParallelTrainer

# assuming there is an environment called 'env'
# and an agent or a list of agents called 'agents'

# create a sequential trainer
cfg = {"timesteps": 50000, "headless": False}
trainer = ParallelTrainer(env=env, agents=agents, cfg=cfg)

# train the agent(s)
trainer.train()

# evaluate the agent(s)
trainer.eval()
# [pytorch-end-parallel]

# =============================================================================

# [pytorch-start-manual]
from skrl.trainers.torch import ManualTrainer

# assuming there is an environment called 'env'
# and an agent or a list of agents called 'agents'

# create a sequential trainer
cfg = {"timesteps": 50000, "headless": False}
trainer = ManualTrainer(env=env, agents=agents, cfg=cfg)

# train the agent(s)
for timestep in range(cfg["timesteps"]):
    trainer.train(timestep=timestep)

# evaluate the agent(s)
for timestep in range(cfg["timesteps"]):
    trainer.eval(timestep=timestep)
# [pytorch-end-manual]


# [jax-start-manual]
from skrl.trainers.jax import ManualTrainer

# assuming there is an environment called 'env'
# and an agent or a list of agents called 'agents'

# create a sequential trainer
cfg = {"timesteps": 50000, "headless": False}
trainer = ManualTrainer(env=env, agents=agents, cfg=cfg)

# train the agent(s)
for timestep in range(cfg["timesteps"]):
    trainer.train(timestep=timestep)

# evaluate the agent(s)
for timestep in range(cfg["timesteps"]):
    trainer.eval(timestep=timestep)
# [jax-end-manual]
