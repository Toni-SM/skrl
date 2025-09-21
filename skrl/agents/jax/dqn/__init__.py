from skrl.agents.jax.dqn.ddqn import DDQN
from skrl.agents.jax.dqn.ddqn_cfg import DDQN_CFG
from skrl.agents.jax.dqn.dqn import DQN
from skrl.agents.jax.dqn.dqn_cfg import DQN_CFG


# fmt: off
from dataclasses import asdict  # isort:skip
DQN_DEFAULT_CONFIG = asdict(DQN_CFG())  # back compatibility config
DDQN_DEFAULT_CONFIG = asdict(DDQN_CFG())  # back compatibility config
# fmt: on
